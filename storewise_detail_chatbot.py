"""
Storewise Sales Dashboard chatbot — natural language -> T-SQL -> result.
Data source: Microsoft SQL Server (live query) via SQLAlchemy + pyodbc.

This is the STOREWISE variant of the sales analytics chatbot. It keeps the retail
SEMANTIC definitions (Trj_Type Sales/Returns, net quantity, net sales, average
per bill, distinct bill counts) but is tuned for a VERY LARGE fact view:

    View name  : vw_storesales_dash
    Row count  : ~20,000,000,000 (20 billion) rows, ~40 columns
    Grain      : one row per line item of a store sale / sale return

The column vocabulary matches `storewise_salesdata_details.xlsx` (the sample
extract of the view), e.g. Doc_Dt, Trj_Type, Pdoc_No, Qty, Rtn_Qty, Net_Amt,
Item, SubGrp, ItemGrp, Scheme, Cobr_Name (store), TotAftDisc_Amt, Taxable_Amt,
Tax_Amt, ...

NET-VALUE RULE (requirements #6 and #7)
    Any "net" figure is computed from Trj_Type as  Sales - Sales Rtn :
      net quantity = SUM(Qty     where Sales) - SUM(Rtn_Qty where Sales Rtn)
      net <amount> = SUM(<amt>    where Sales) - SUM(<amt>    where Sales Rtn)
    (returns are stored as POSITIVE values on the 'Sales Rtn' row, so they are
    subtracted, never added.) This generalises to *every* amount column
    (Net_Amt, Taxable_Amt, Tax_Amt, TotAftDisc_Amt, ...), not just net sales.

BIG-DATA / SPEED (requirements #4 and #5)
    Retrieval is optimised for 20B rows. See PERFORMANCE NOTES lower down; the
    short version:
      * Every query is server-side AGGREGATED — raw rows are never streamed to
        the client (a guard rejects un-aggregated, un-TOP'd scans).
      * Every query is DATE-BOUNDED on Doc_Dt so the engine can do partition
        elimination; if the user names no period, a default recent window is
        injected automatically.
      * Connections run with READ UNCOMMITTED + SET NOCOUNT/ARITHABORT and a
        statement timeout, the standard read-only reporting profile.
      * Bill counts default to APPROX_COUNT_DISTINCT (SQL Server 2019+), which is
        dramatically cheaper than exact COUNT(DISTINCT) at this scale. Flip the
        sidebar toggle for exact counts.
      * Identical questions are served from a short-lived result cache instead of
        re-hitting the database.

Features carried over unchanged: voice input (Groq Whisper), hierarchical
sub-totals + grand total, whole-number formatting, proper-case labels, hidden
SQL (kept in an expander for debugging), searchable prompt history.

Note: the mic needs Streamlit >= 1.36 and a secure context (https:// or
localhost).

Setup:
    pip install -r requirements.txt
    # requires a SQL Server ODBC driver on the host, e.g.
    #   "ODBC Driver 17 for SQL Server" or "ODBC Driver 18 for SQL Server"
    # put your secrets in .streamlit/secrets.toml:
    #   GROQ_API_KEY   = "gsk_..."
    #   HS_DB_SERVER   = "..."
    #   HS_DB_PORT     = 1433
    #   HS_DB_DATABASE = "..."
    #   HS_DB_USERNAME = "..."
    #   HS_DB_PASSWORD = "..."
    #   HS_DB_DRIVER   = "ODBC Driver 18 for SQL Server"
    streamlit run storesales_dash_chatbot.py
"""

import re
import time
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from groq import Groq

# Name of the SQL Server view that holds the storewise sales rows.
TABLE = "vw_storesales_dash"
GROQ_MODEL = "openai/gpt-oss-120b"   # VERIFY current IDs at console.groq.com/docs/models
# Speech-to-text (voice input). Groq's ASR API is OpenAI-compatible; turbo is the
# fastest/cheapest Whisper. Alternatives: "whisper-large-v3", "distil-whisper-large-v3-en".
GROQ_STT_MODEL = "whisper-large-v3-turbo"

# ---------------------------------------------------------------------------
# PERFORMANCE NOTES  (why this file is shaped the way it is, for ~20B rows)
# ---------------------------------------------------------------------------
# The view is assumed to sit on a CLUSTERED COLUMNSTORE INDEX partitioned by
# Doc_Dt (month/day). That is the single most important thing for this workload:
# columnstore gives batch-mode aggregation + segment elimination, and date
# partitioning lets a Doc_Dt filter skip almost every partition. Nothing in this
# app can create those indexes — they belong to the DBA — but the app is written
# to *exploit* them:
#
#   1. AGGREGATE-ONLY. Every question is answered with GROUP BY + SUM/approx
#      counts, so only a tiny result set crosses the wire. Un-aggregated,
#      un-TOP'd "give me the rows" scans are rejected before they ever run.
#   2. ALWAYS DATE-BOUNDED. A Doc_Dt range is required so the optimizer can do
#      partition elimination. If the user doesn't give one, DEFAULT_LOOKBACK_DAYS
#      is injected and the UI says so.
#   3. READ UNCOMMITTED session. Reporting reads don't take shared locks, so they
#      don't fight the OLTP writers loading the view. Plus SET NOCOUNT ON /
#      ARITHABORT ON and a hard statement timeout.
#   4. APPROX_COUNT_DISTINCT for bill counts by default — ~2% error, a fraction
#      of the memory/time of exact COUNT(DISTINCT) over billions of rows.
#   5. RESULT CACHE. Identical SQL is served from an in-process cache (TTL) so a
#      dashboard that re-asks the same thing doesn't re-scan the view.
#   6. NO SELECT *. Only the grouped dimensions + measures are ever projected.
# ---------------------------------------------------------------------------
DEFAULT_LOOKBACK_DAYS = 30      # window injected when the user names no period
QUERY_TIMEOUT_S       = 300     # statement timeout; kill runaway scans
RESULT_CACHE_TTL_S    = 300     # how long identical questions are cached
MAX_RESULT_ROWS       = 100_000 # client-side safety net on rows fetched
READ_UNCOMMITTED      = True    # standard read-only reporting isolation

# ---------------------------------------------------------------------------
# Database connection (Microsoft SQL Server via SQLAlchemy + pyodbc)
# Credentials live in .streamlit/secrets.toml.
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "server":   st.secrets["HS_DB_SERVER"],
    "port":     st.secrets["HS_DB_PORT"],
    "database": st.secrets["HS_DB_DATABASE"],
    "username": st.secrets["HS_DB_USERNAME"],
    "password": st.secrets["HS_DB_PASSWORD"],
    "driver":   st.secrets["HS_DB_DRIVER"],
}

@st.cache_resource
def get_engine():
    """Create a cached SQLAlchemy engine for the remote MSSQL server.

    Uses st.cache_resource (not cache_data) so the connection pool is reused
    across reruns instead of being re-created every time.

    Every new physical connection is put into the read-only reporting profile
    (READ UNCOMMITTED, NOCOUNT, ARITHABORT) and given a statement timeout, so no
    single query can hold the app hostage while scanning the 20B-row view.
    """
    url = URL.create(
        "mssql+pyodbc",
        username=DB_CONFIG["username"],
        password=DB_CONFIG["password"],   # URL.create escapes special chars safely
        host=DB_CONFIG["server"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        query={
            "driver": DB_CONFIG["driver"],
            "TrustServerCertificate": "yes",   # needed for most internal/self-signed servers
            "Encrypt": "no",
        },
    )
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,      # recycle idle conns so long-lived dashboards stay healthy
        pool_size=5,
        max_overflow=10,
    )

    @event.listens_for(engine, "connect")
    def _init_connection(dbapi_conn, conn_record):
        # pyodbc: cap how long any single statement may run.
        try:
            dbapi_conn.timeout = QUERY_TIMEOUT_S
        except Exception:
            pass
        cur = dbapi_conn.cursor()
        # One batch of session settings, applied once per physical connection:
        #  - NOCOUNT: skip per-statement row-count messages (less chatter).
        #  - ARITHABORT ON: recommended for consistent, plan-friendly execution.
        #  - READ UNCOMMITTED: reporting reads take no shared locks -> they don't
        #    block, and aren't blocked by, the writers loading the view.
        settings = "SET NOCOUNT ON; SET ARITHABORT ON;"
        if READ_UNCOMMITTED:
            settings += " SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;"
        cur.execute(settings)
        cur.close()

    return engine

def _execute(sql: str) -> pd.DataFrame:
    """Execute a read-only SELECT and return a DataFrame (no caching).

    exec_driver_sql sends the string straight to the DBAPI, so colons in literals
    aren't mistaken for bind parameters. We fetch at most MAX_RESULT_ROWS as a
    client-side safety net — legitimate aggregated queries return far fewer.
    """
    with get_engine().connect() as conn:
        res = conn.exec_driver_sql(sql)
        rows = res.fetchmany(MAX_RESULT_ROWS)
        cols = list(res.keys())
    df = pd.DataFrame.from_records(rows, columns=cols)
    # pyodbc returns SQL Server decimal/money/numeric as Python Decimal objects,
    # which land in pandas as `object` dtype and are NOT seen as numeric — so the
    # chart logic mistakes a measure (e.g. SUM(Net_Amt)) for a dimension. Convert
    # any object column that is fully numeric back to real numbers.
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            # only replace if every non-null original value converted cleanly
            # (leaves genuine text dimensions like Item/Cobr_Name untouched)
            if df[c].notna().any() and conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df

@st.cache_data(ttl=RESULT_CACHE_TTL_S, show_spinner=False)
def run_query_cached(sql: str) -> pd.DataFrame:
    """Result-cached wrapper around _execute for user-facing analytics queries.

    Two dashboard users (or the same user re-asking) with the identical generated
    SQL get an instant answer from cache instead of a fresh 20B-row aggregation.
    The cache is keyed on the exact SQL string, so any change in period, measure,
    grouping, or the approx/exact-count toggle produces a distinct key.
    """
    return _execute(sql)

def run_query(sql: str) -> pd.DataFrame:
    """Uncached read — used for lightweight metadata (introspection) only."""
    return _execute(sql)

@st.cache_data(ttl=3600, show_spinner=False)
def get_table_columns() -> list[str]:
    """Introspect the actual columns of the view so the model only ever
    references real column names. Cached for an hour — the schema is stable."""
    sql = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        f"WHERE TABLE_NAME = '{TABLE}' ORDER BY ORDINAL_POSITION"
    )
    df = run_query(sql)
    return df["COLUMN_NAME"].tolist() if not df.empty else []

# ---------------------------------------------------------------------------
# Performance controls read from the sidebar (with safe defaults). Placed here
# so the semantic layer / prompt can consult them each turn.
# ---------------------------------------------------------------------------
def _exact_counts() -> bool:
    """True -> exact COUNT(DISTINCT Pdoc_No); False -> APPROX_COUNT_DISTINCT."""
    return bool(st.session_state.get("exact_counts", False))

def _lookback_days() -> int:
    return int(st.session_state.get("lookback_days", DEFAULT_LOOKBACK_DAYS))

def _default_period():
    """(start_iso, end_iso) window injected when the user names no period."""
    end = date.today()
    start = end - timedelta(days=_lookback_days())
    return start.isoformat(), end.isoformat()

# ---------------------------------------------------------------------------
# Semantic layer  (storewise retail meaning, expressed as T-SQL)
# ---------------------------------------------------------------------------
SEMANTIC_LAYER_TMPL = """
View `{table}` — one row per line item of a store sale or sale return. This is a
VERY LARGE fact view (~20 billion rows). Queries MUST be cheap: always aggregate,
always bound by Doc_Dt.

Key columns (fixed business meaning):
- Doc_Dt        : business/transaction date. USE THIS for any period, "per day",
                  "monthly", "over time", or date-range filter. Time part is
                  always 00:00:00, so CAST(Doc_Dt AS DATE) is the day key.
- Created_Dt / Created_Time : system clock timestamp. Do NOT use for
                  business-day analysis; only for "what time of day" questions.
- Trj_Type      : 'Sales' (a sale) or 'Sales Rtn' (a return). CRITICAL for every
                  net figure.
- Pdoc_No       : bill / transaction number. One bill spans many line rows, so a
                  COUNT of transactions/bills/orders must count DISTINCT Pdoc_No,
                  never COUNT(*).
- Qty           : sold quantity. Positive on 'Sales' rows, 0 on 'Sales Rtn' rows.
- Rtn_Qty       : returned quantity. POSITIVE on 'Sales Rtn' rows, 0 on 'Sales' rows.
- Net_Amt       : line net amount. The default revenue measure unless the user
                  names another. On a 'Sales Rtn' row it holds the (positive)
                  return value.
- Amount columns (all positive on their row; subtract the return part for a net):
                  Net_Amt, TotAftDisc_Amt, Taxable_Amt, TotOthDisc_Amt, Tax_Amt,
                  LandedCostIncl_Tax, ItemDisc_Amt, Drs_Amt.
- Cobr_Name     : STORE name (this is a storewise dashboard — "by store", "per
                  store", "which store" all group by Cobr_Name).
- Item, SubGrp, ItemGrp, Scheme, Doc_Type, Ean_Barcode : dimensions.
- Other dimensions if present: Supplier, SalePerson, Location, State, Country,
                  Station, User_Name, Ctr_Name, Party.

The full, authoritative column list for `{table}` is:
{columns}
Use those exact names. NEVER invent a column that is not in that list.

NET-VALUE RULE (Sales - Sales Rtn) — applies to ANY net figure:
- "net quantity" / "net qty"  (units sold minus units returned)
      -> SUM(CASE WHEN Trj_Type = 'Sales' THEN Qty ELSE 0 END)
         - SUM(CASE WHEN Trj_Type = 'Sales Rtn' THEN Rtn_Qty ELSE 0 END)
- "net sales" / "net revenue" / "net <amount>"  (value sold minus value returned)
      -> SUM(CASE WHEN Trj_Type = 'Sales' THEN <AMT_COL> ELSE 0 END)
         - SUM(CASE WHEN Trj_Type = 'Sales Rtn' THEN <AMT_COL> ELSE 0 END)
   where <AMT_COL> is Net_Amt for "net sales", or the named amount column for any
   other net figure (e.g. "net taxable" -> Taxable_Amt, "net tax" -> Tax_Amt).
   Returns are stored POSITIVE, so they are SUBTRACTED, never added.

GROSS METRIC CONVENTIONS (follow exactly; these are different queries):
- "quantity" / "qty" / "sold" / "selling"  -> SUM(Qty) WHERE Trj_Type = 'Sales'
- "sales" / "revenue" / "sales value"      -> SUM(Net_Amt) WHERE Trj_Type = 'Sales'
- "returns" / "return qty"                 -> SUM(Rtn_Qty) WHERE Trj_Type = 'Sales Rtn'
- "return value"                           -> SUM(Net_Amt) WHERE Trj_Type = 'Sales Rtn'
Use SUM by default. Only average when the user says "average".
- "average sales" / "average bill" (per BILL) -> SUM(Net_Amt) / {count_distinct}
      NOT AVG(Net_Amt): that averages line items and is skewed by single large lines.
- "how many" / "count of" bills/transactions/orders/invoices
      -> {count_distinct}   (rows are line items, not bills)
Default to sales quantity (SUM(Qty), Sales-only) for "top/best/highest selling".

DATES & PERIODS (mandatory — the view is 20B rows):
- Dates in questions are day-first: dd/MM/yyyy. So 03/04/2024 = 3 April 2024.
- EVERY query MUST filter Doc_Dt so the engine can eliminate partitions.
- "for the period X to Y" -> CAST(Doc_Dt AS DATE) BETWEEN 'yyyy-mm-dd' AND 'yyyy-mm-dd'
- If the user names NO period, use the default recent window:
      CAST(Doc_Dt AS DATE) BETWEEN '{def_start}' AND '{def_end}'
- For "per month" / "monthly", group by FORMAT(Doc_Dt, 'yyyy-MM').
"""

def _count_distinct_expr() -> str:
    """Exact vs approximate distinct-bill expression, per the sidebar toggle."""
    if _exact_counts():
        return "COUNT(DISTINCT Pdoc_No)"
    return "APPROX_COUNT_DISTINCT(Pdoc_No)"   # SQL Server 2019+; ~2% error, far cheaper

FEW_SHOTS_TMPL = """
Q: Net sales by store for the period 01/07/2025 to 31/07/2025
SQL: SELECT Cobr_Name,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN Net_Amt ELSE 0 END)
            - SUM(CASE WHEN Trj_Type = 'Sales Rtn' THEN Net_Amt ELSE 0 END) AS net_sales
     FROM {table}
     WHERE CAST(Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
     GROUP BY Cobr_Name ORDER BY net_sales DESC;

Q: Top 10 items by quantity for the period 01/07/2025 to 31/07/2025
SQL: SELECT TOP 10 Item, SUM(Qty) AS qty
     FROM {table}
     WHERE Trj_Type = 'Sales'
       AND CAST(Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
     GROUP BY Item ORDER BY qty DESC;

Q: Top 10 item groups by net quantity for the period 01/07/2025 to 31/07/2025
SQL: SELECT TOP 10 ItemGrp,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN Qty ELSE 0 END)
            - SUM(CASE WHEN Trj_Type = 'Sales Rtn' THEN Rtn_Qty ELSE 0 END) AS net_qty
     FROM {table}
     WHERE CAST(Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
     GROUP BY ItemGrp ORDER BY net_qty DESC;

Q: Net sales by store and item group for the period 01/07/2025 to 31/07/2025
SQL: SELECT Cobr_Name, ItemGrp,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN Net_Amt ELSE 0 END)
            - SUM(CASE WHEN Trj_Type = 'Sales Rtn' THEN Net_Amt ELSE 0 END) AS net_sales
     FROM {table}
     WHERE CAST(Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
     GROUP BY Cobr_Name, ItemGrp ORDER BY Cobr_Name, net_sales DESC;

Q: Net quantity by sub group for the top 5 stores for the period 01/07/2025 to 31/07/2025
SQL: WITH top_stores AS (
       SELECT TOP 5 Cobr_Name
       FROM {table}
       WHERE CAST(Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
       GROUP BY Cobr_Name
       ORDER BY SUM(CASE WHEN Trj_Type = 'Sales' THEN Qty ELSE 0 END)
                - SUM(CASE WHEN Trj_Type = 'Sales Rtn' THEN Rtn_Qty ELSE 0 END) DESC
     )
     SELECT s.Cobr_Name, s.SubGrp,
            SUM(CASE WHEN s.Trj_Type = 'Sales' THEN s.Qty ELSE 0 END)
            - SUM(CASE WHEN s.Trj_Type = 'Sales Rtn' THEN s.Rtn_Qty ELSE 0 END) AS net_qty
     FROM {table} s
     WHERE CAST(s.Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
       AND s.Cobr_Name IN (SELECT Cobr_Name FROM top_stores)
     GROUP BY s.Cobr_Name, s.SubGrp ORDER BY s.Cobr_Name, net_qty DESC;

Q: How many bills per store for the period 01/07/2025 to 31/07/2025
SQL: SELECT Cobr_Name, {count_distinct} AS bills
     FROM {table}
     WHERE Trj_Type = 'Sales'
       AND CAST(Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
     GROUP BY Cobr_Name ORDER BY bills DESC;

Q: Average bill value by store for the period 01/07/2025 to 31/07/2025
SQL: SELECT Cobr_Name, SUM(Net_Amt) / {count_distinct} AS avg_bill
     FROM {table}
     WHERE Trj_Type = 'Sales'
       AND CAST(Doc_Dt AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'
     GROUP BY Cobr_Name ORDER BY avg_bill DESC;

Q: Net sales per month for the period 01/01/2025 to 30/06/2025
SQL: SELECT FORMAT(Doc_Dt, 'yyyy-MM') AS month,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN Net_Amt ELSE 0 END)
            - SUM(CASE WHEN Trj_Type = 'Sales Rtn' THEN Net_Amt ELSE 0 END) AS net_sales
     FROM {table}
     WHERE CAST(Doc_Dt AS DATE) BETWEEN '2025-01-01' AND '2025-06-30'
     GROUP BY FORMAT(Doc_Dt, 'yyyy-MM') ORDER BY month;

Q: Top selling item (no period given)
SQL: SELECT TOP 1 Item, SUM(Qty) AS qty
     FROM {table}
     WHERE Trj_Type = 'Sales'
       AND CAST(Doc_Dt AS DATE) BETWEEN '{def_start}' AND '{def_end}'
     GROUP BY Item ORDER BY qty DESC;
"""

def build_system_prompt() -> str:
    cols = get_table_columns()
    col_block = ", ".join(cols) if cols else "(column list unavailable)"
    def_start, def_end = _default_period()
    cnt = _count_distinct_expr()
    semantic = SEMANTIC_LAYER_TMPL.format(
        table=TABLE, columns=col_block,
        count_distinct=cnt, def_start=def_start, def_end=def_end,
    )
    few_shots = FEW_SHOTS_TMPL.format(
        table=TABLE, count_distinct=cnt, def_start=def_start, def_end=def_end,
    )
    return f"""You translate a question about storewise retail sales into ONE Microsoft
SQL Server (T-SQL) SELECT statement. Output ONLY the SQL — no prose, no markdown fences.

{semantic}

Examples:
{few_shots}

Rules:
- This is T-SQL (Microsoft SQL Server). Use SELECT TOP N for "top/highest",
  NOT LIMIT. There is no LIMIT clause in SQL Server.
- Exactly one statement, must start with SELECT or WITH.
- Never INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/EXEC/MERGE/GRANT.
- The view has ~20 BILLION rows. EVERY query MUST:
    (a) AGGREGATE — use GROUP BY with SUM / the distinct-bill expression; never
        return raw line-item rows and never SELECT *.
    (b) FILTER Doc_Dt with a date range (use the default window if the user gives
        no period), so the engine can eliminate partitions.
- For counts of bills/transactions use exactly: {cnt}
- For any NET figure use the Sales - Sales Rtn CASE formula shown above.
- Always add a sensible TOP N for "top/highest/best" style questions.
- When the user names two dimensions (e.g. "by store and item group"), GROUP BY
  both and SELECT both plus the measure, so the result draws as a stacked chart.
- CRITICAL for a "top N" with two dimensions: apply TOP to the PRIMARY dimension
  via a CTE (top N of dim1 by the measure), then return ALL rows of dim2 for
  those. NEVER put TOP on the (dim1, dim2) combination.
- Group by the date EXPRESSION, not its alias (T-SQL can't GROUP BY an alias):
  e.g. GROUP BY CAST(Doc_Dt AS DATE), not GROUP BY day.
- Reference ONLY columns from the column list above. If the question is
  impossible with the schema, return exactly: SELECT 'unanswerable' AS note;
"""

# ---------------------------------------------------------------------------
# SQL guardrail
# ---------------------------------------------------------------------------
_BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|exec|execute|merge|"
    r"grant|revoke|backup|restore|shutdown|waitfor|attach|copy|pragma|call|export)\b",
    re.IGNORECASE,
)
_AGG_FN = re.compile(
    r"\b(sum|count|avg|min|max|approx_count_distinct)\s*\(", re.IGNORECASE
)

def _is_aggregated(sql: str) -> bool:
    """A query is 'safe at scale' if it aggregates (GROUP BY or an aggregate
    function) or is a bounded TOP N — anything else risks streaming billions of
    raw rows and is rejected."""
    if re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE):
        return True
    if _AGG_FN.search(sql):
        return True
    if re.search(r"\bselect\s+top\s+\d+\b", sql, re.IGNORECASE):
        return True
    # the model's explicit "can't answer" escape hatch
    if re.search(r"'unanswerable'", sql, re.IGNORECASE):
        return True
    return False

def _has_date_filter(sql: str) -> bool:
    """True if the query bounds Doc_Dt (partition elimination). Accepts BETWEEN,
    comparison operators, or common date functions applied to Doc_Dt."""
    return bool(re.search(r"doc_dt", sql, re.IGNORECASE)) and bool(
        re.search(r"\bwhere\b", sql, re.IGNORECASE)
    )

def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise ValueError("Query must start with SELECT or WITH.")
    if _BLOCKED.search(sql):
        raise ValueError("Query contains a forbidden keyword.")
    # Big-data safety net: never let an un-aggregated scan hit the 20B-row view.
    if not re.search(r"'unanswerable'", sql, re.IGNORECASE):
        if not _is_aggregated(sql):
            raise ValueError(
                "That would scan raw rows of a 20-billion-row view. Please ask "
                "for an aggregate (a total, a count, or a top-N), e.g. "
                "'net sales by store this month'."
            )
        if not _has_date_filter(sql):
            raise ValueError(
                "Queries on this view must be bounded by a date range on Doc_Dt. "
                "Please include a period, e.g. 'for the period 01/07/2025 to "
                "31/07/2025', or just 'this month'."
            )
    return sql

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def _extract_sql(text: str) -> str:
    """Pull a single SQL statement out of the model reply, tolerating a
    reasoning preamble, code fences, or trailing prose."""
    if not text:
        raise ValueError("Empty response from model.")
    # prefer a fenced code block if present
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    # take everything from the first SELECT / WITH onward (drops any preamble)
    m = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    # cut at the first semicolon (drops any trailing explanation)
    return text.split(";")[0].strip()

def generate_sql(question: str) -> str:
    resp = get_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=2048,            # headroom so reasoning can't crowd out the SQL
        reasoning_effort="low",     # gpt-oss: minimize reasoning for this simple task
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ],
    )
    msg = resp.choices[0].message
    # content is normally the SQL; if empty (reasoning used the budget), the SQL
    # is almost always still inside the reasoning text — fall back to that.
    text = (msg.content or "").strip() or (getattr(msg, "reasoning", "") or "").strip()
    return validate_sql(_extract_sql(text))

# ---------------------------------------------------------------------------
# Speech-to-text (voice input)
#   Sends recorded microphone audio to Groq's Whisper endpoint and returns the
#   transcribed text. Reuses the same Groq client / GROQ_API_KEY as the SQL
#   generation above, so no extra credentials or packages are required.
# ---------------------------------------------------------------------------
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe WAV audio bytes (from st.audio_input) to text via Groq Whisper."""
    resp = get_client().audio.transcriptions.create(
        file=("voice.wav", audio_bytes),   # (filename, bytes) — st.audio_input yields WAV
        model=GROQ_STT_MODEL,
        response_format="text",            # returns a plain string, not JSON
        # language="en",                   # uncomment to force English recognition
    )
    # With response_format="text" the SDK returns a str; be defensive either way.
    return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()

# ---------------------------------------------------------------------------
# Charting
#   1 measure + 1 dimension  -> bar, ordered by value, data labels
#   1 measure + 2 dimensions -> STACKED bar (2nd dim = color) with legend
# ---------------------------------------------------------------------------
# Column display names shown on charts (axis titles, legend, hover) and in
# table headers. Raw SQL aliases / column names are converted to proper case.
def _pretty(name: str) -> str:
    """Return a proper-case, human-friendly label for a column / alias."""
    return str(name).replace("_", " ").title()

def _round_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Round every numeric measure to a whole number (no decimals shown)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(0).astype("Int64")   # nullable int keeps NaNs
    return out

def _split_cols(df: pd.DataFrame):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dim = [c for c in df.columns if c not in num]
    return num, dim

def chartable(df: pd.DataFrame) -> bool:
    num, dim = _split_cols(df)
    return len(num) == 1 and len(dim) in (1, 2) and len(df) > 0

def render_chart(df: pd.DataFrame, key: str = None):
    num, dim = _split_cols(df)
    measure = num[0]
    d = df.copy()
    for c in dim:                       # dims must be categorical strings so
        d[c] = d[c].fillna("Unknown").astype(str)   # plotly stacks them
    x = dim[0]
    order = (d.groupby(x)[measure].sum()
               .sort_values(ascending=False).index.tolist())
    labels = {c: _pretty(c) for c in d.columns}   # proper-case display names
    if len(dim) == 1:
        fig = px.bar(d, x=x, y=measure, text_auto=",.0f",   # data labels: no decimals
                     category_orders={x: order}, labels=labels)
        fig.update_traces(textposition="outside")
    else:
        color = dim[1]
        fig = px.bar(d, x=x, y=measure, color=color, barmode="stack",
                     text_auto=",.0f", category_orders={x: order}, labels=labels)
        fig.update_layout(legend_title_text=_pretty(color))   # proper-case legend title
    fig.update_layout(
        xaxis_title=_pretty(x), yaxis_title=_pretty(measure),
        height=650, margin=dict(l=50, r=20, t=30, b=50),
        showlegend=True,                 # remove legend (incl. 2-dimension charts)
        font=dict(color="black"),         # axis titles + tick labels in black
    )
    fig.update_yaxes(tickformat=",.0f")   # measures: whole numbers, no decimals
    st.plotly_chart(fig, use_container_width=True, key=key)

def _show_table(df: pd.DataFrame):
    """Display a table with a 1-based Sr. No column instead of the 0-based index.
    Measures are rounded to whole numbers and headers shown in proper case."""
    disp = _round_measures(df)                                  # no decimals on measures
    disp = disp.rename(columns={c: _pretty(c) for c in disp.columns})   # proper-case headers
    disp.insert(0, "Sr. No", range(1, len(disp) + 1))
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Totals for the "Table only" view
#   1 dimension    -> a Grand Total row.
#   2+ dimensions  -> HIERARCHICAL sub-totals: a sub-total after each group at
#                     EVERY grouping level, then a Grand Total. e.g. grouping by
#                     Store > SubGrp > Item gives an Item block per SubGrp, a
#                     SubGrp sub-total, a Store sub-total (over its SubGrps), and
#                     finally the Grand Total. Each group therefore gets its own
#                     total at its own level.
# Only additive measures (SUM of qty / amount / counts) are totalled. A measure
# whose name implies an average or ratio (e.g. avg_bill) can't be summed
# meaningfully, so its total cells are left blank instead of showing a wrong sum.
# ---------------------------------------------------------------------------
_NON_ADDITIVE = ("avg", "average", "mean", "ratio", "per ")

# total_level tags on each output row, consumed by the table styler:
#   DETAIL_LEVEL -> a normal data row (no shading)
#   0, 1, 2, ... -> a sub-total for the dimension at that index (0 = first dim)
#   GRAND_LEVEL  -> the single Grand Total row
DETAIL_LEVEL = -1
GRAND_LEVEL = -2

# Row background per total level: Grand Total darkest, then progressively lighter
# for deeper (higher-index) sub-totals so the hierarchy reads at a glance.
_GRAND_SHADE = "#cdd7e6"
_SUBTOTAL_SHADES = ["#dbe2ee", "#e4eaf3", "#eaeff6", "#eff2f7"]

def _total_shade(level: int) -> str:
    if level == GRAND_LEVEL:
        return _GRAND_SHADE
    return _SUBTOTAL_SHADES[min(level, len(_SUBTOTAL_SHADES) - 1)]

def _is_additive(measure: str) -> bool:
    k = str(measure).lower()
    return not any(t in k for t in _NON_ADDITIVE)

def _sum_measures(block: pd.DataFrame, measures: list) -> dict:
    """SUM each additive measure over `block`; blank (NA) for non-additive ones."""
    return {m: (block[m].sum() if _is_additive(m) else pd.NA) for m in measures}

def _build_totals(df: pd.DataFrame, dims: list, measures: list):
    """Return (augmented_df, is_total_mask, total_level).

    Detail rows keep their original (SQL) order within their block. For 2+
    dimensions the data is nested hierarchically and a sub-total row is emitted
    at EVERY grouping level: after each innermost block, then after its parent,
    up to the first dimension, and finally one Grand Total. Groups at each level
    are ordered biggest-first by the first additive measure, so every block sits
    directly under its own rows even when the SQL interleaved them.

    total_level[i] tags row i: DETAIL_LEVEL for data rows, the dimension index
    (0 = first dim) for a sub-total at that level, GRAND_LEVEL for the grand
    total — used by the styler to shade each total by its depth."""
    d = df.copy()
    for c in dims:
        d[c] = d[c].astype(object)          # allow string labels / blanks in dim cells
    rows, is_total, total_level = [], [], []

    def _ordered_values(block, dim):
        """Distinct values of `dim` in `block`, biggest additive measure first."""
        additive = [m for m in measures if _is_additive(m)]
        if additive:
            return (block.groupby(dim)[additive[0]].sum()
                         .sort_values(ascending=False).index.tolist())
        return list(pd.unique(block[dim]))    # no additive measure -> first-seen order

    def emit_detail(block):
        for _, r in block.iterrows():
            rows.append(r.to_dict()); is_total.append(False)
            total_level.append(DETAIL_LEVEL)

    def recurse(block, level):
        """Walk one grouping level: recurse into children, then sub-total this
        group. The last dimension is the leaf whose rows are the detail rows."""
        dim = dims[level]
        is_leaf = (level == len(dims) - 1)
        for v in _ordered_values(block, dim):
            sub_block = block[block[dim] == v]
            if is_leaf:
                emit_detail(sub_block)
            else:
                recurse(sub_block, level + 1)
                sub = {c: "" for c in d.columns}
                sub[dim] = f"{v} \u2014 Total"        # e.g. "STORE-01 — Total"
                sub.update(_sum_measures(sub_block, measures))
                rows.append(sub); is_total.append(True)
                total_level.append(level)

    if len(dims) == 1:
        emit_detail(d)
    else:
        recurse(d, 0)

    grand = {c: "" for c in d.columns}
    grand[dims[0]] = "Grand Total"
    grand.update(_sum_measures(d, measures))
    rows.append(grand); is_total.append(True); total_level.append(GRAND_LEVEL)

    out = pd.DataFrame(rows, columns=d.columns)
    for m in measures:                        # whole numbers, keep <NA> for ratios
        out[m] = pd.to_numeric(out[m], errors="coerce").round(0).astype("Int64")
    return out, is_total, total_level

def _show_table_with_totals(df: pd.DataFrame, dims: list, measures: list, key=None):
    """Table view with sub-total / grand-total rows, shaded and bold."""
    aug, is_total, total_level = _build_totals(df, dims, measures)
    disp = aug.rename(columns={c: _pretty(c) for c in aug.columns})
    # Sr. No: number the detail rows only; total rows (any level) get a blank.
    srno, n = [], 0
    for t in is_total:
        if t:
            srno.append("")
        else:
            n += 1
            srno.append(n)
    disp.insert(0, "Sr. No", srno)

    measure_labels = [_pretty(m) for m in measures]

    def _highlight(row):
        lvl = total_level[row.name]
        if lvl == DETAIL_LEVEL:
            return [""] * len(row)
        bg = _total_shade(lvl)          # deeper sub-total -> lighter; grand -> darkest
        return [f"background-color: {bg}; font-weight: bold;"] * len(row)

    styler = (disp.style
                  .apply(_highlight, axis=1)
                  .format("{:,.0f}", subset=measure_labels, na_rep=""))  # blank NA totals
    st.dataframe(styler, use_container_width=True, hide_index=True, key=key)

# ---------------------------------------------------------------------------
# Per-prompt timing
#   Wall-clock timings shown under each answer so you can see where the time
#   goes: SQL generation (the LLM call) vs. query execution (the database), plus
#   the total. Note: a repeated question is served from run_query_cached, so its
#   "query" time will be a few ms — flip "Bypass result cache" in the sidebar to
#   always measure the raw database round-trip.
# ---------------------------------------------------------------------------
def _fmt_secs(s: float) -> str:
    """Human-friendly duration: ms under a second, else seconds to 2 dp."""
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s"

def _timing_caption(t: dict) -> str:
    parts = []
    if "sql_gen" in t:
        parts.append(f"SQL gen {_fmt_secs(t['sql_gen'])}")
    if "query" in t:
        q = f"query {_fmt_secs(t['query'])}"
        if t.get("cached"):
            q += " (cached)"
        parts.append(q)
    if "total" in t:
        parts.append(f"total {_fmt_secs(t['total'])}")
    return "\u23f1\ufe0f " + "  \u00b7  ".join(parts)   # ⏱️ a · b · c

def show_result(table: pd.DataFrame, sql: str, mode: str, key: str = None):
    # SQL is hidden from the UI — results only. (sql is still stored in
    # session_state for history/debugging; the expander shows it on demand.)
    with st.expander("SQL"):
        st.code(sql, language="sql")
    can = chartable(table)
    num, dim = _split_cols(table)       # measures, dimensions
    if len(table) <= 1:                 # single result -> table only, no graph
        _show_table(table)
        return
    if mode == "Table only":
        # 1 dim -> grand total; 2+ dims -> sub-totals + grand total.
        if len(dim) >= 1 and len(num) >= 1:
            _show_table_with_totals(table, dim, num, key=key)
        else:                            # no groupable dimension/measure -> plain
            _show_table(table)
    elif mode == "Graph only":
        if can:
            render_chart(table, key=key)
        else:
            st.info("This result can't be charted — showing the table instead.")
            _show_table(table)
    else:  # Table + Graph
        _show_table(table)
        if can:
            render_chart(table, key=key)

# ---------------------------------------------------------------------------
# Sticky header CSS — pins the title + display selector to the top of the page.
# Uses the :has() trick to make the header container sticky. Adjust `top` if
# it overlaps the Streamlit toolbar on your version.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Storewise Sales Dashboard Chatbot", layout="wide")
st.markdown("""
<style>
  /* tighten page whitespace */
  div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
  /* smaller vertical gaps between elements */
  div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
  /* nominal chat input height */
  div[data-testid="stChatInput"] textarea { min-height: 2.2rem; }
  /* tighter chat message bubbles */
  div[data-testid="stChatMessage"] { padding: 0.35rem 0.6rem; }
  /* sticky header, no border */
  div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
      position: sticky; top: 0; z-index: 999;
      background-color: var(--background-color, white);
  }
</style>
""", unsafe_allow_html=True)

header = st.container()
header.markdown(
    "<h1 style='font-size:1.6rem; margin:0 0 0.3rem 0;'>Storewise Sales Dashboard Chatbot</h1>",
    unsafe_allow_html=True,
)
c1, c2 = header.columns([1, 10], vertical_alignment="center")
c1.markdown("**Display**")
view_mode = c2.radio(
    "Display", ["Table + Graph", "Table only", "Graph only"],
    index=1, horizontal=True, label_visibility="collapsed", key="view_mode",
)
header.markdown('<div class="fixed-header"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — performance controls + searchable prompt history.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Performance")
    st.caption(
        f"View `{TABLE}` (~20B rows). Every query is aggregated, date-bounded, "
        "and result-cached for speed."
    )
    st.slider(
        "Default period (days back)", min_value=1, max_value=365,
        value=DEFAULT_LOOKBACK_DAYS, key="lookback_days",
        help="Window applied automatically when you don't name a period.",
    )
    st.toggle(
        "Exact bill counts (slower)", value=False, key="exact_counts",
        help="Off: APPROX_COUNT_DISTINCT (fast, ~2% error). "
             "On: exact COUNT(DISTINCT Pdoc_No).",
    )
    st.toggle(
        "Bypass result cache (measure raw DB time)", value=False,
        key="bypass_cache",
        help="On: every question hits the database directly, so the per-prompt "
             "timer shows the true query time instead of a cache hit.",
    )
    st.divider()

    st.subheader("Prompt history")
    search = st.text_input("Search", placeholder="filter prompts...")
    user_msgs = [(i, m["content"]) for i, m in enumerate(st.session_state.messages)
                 if m["role"] == "user"]
    if not user_msgs:
        st.caption("No prompts yet.")
    for i, text in reversed(user_msgs):          # newest first
        if search and search.lower() not in text.lower():
            continue
        label = text if len(text) <= 45 else text[:42] + "..."
        safe = text.replace('"', "&quot;")
        st.markdown(f'<a href="#msg{i}" title="{safe}">{label}</a>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Conversation — anchor before each user turn so the sidebar can scroll to it.
# ---------------------------------------------------------------------------
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        st.markdown(f"<div id='msg{i}'></div>", unsafe_allow_html=True)
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("timing"):
            st.caption(_timing_caption(m["timing"]))
        if m.get("table") is not None:
            show_result(m["table"], m["sql"], view_mode, key=f"hist_{i}")

def handle_prompt(prompt: str):
    """Run one turn: record the user prompt, generate SQL, query, render.
    Shared by both typed (chat_input) and spoken (voice) prompts."""
    idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div id='msg{idx}'></div>", unsafe_allow_html=True)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            bypass = bool(st.session_state.get("bypass_cache"))
            read = run_query if bypass else run_query_cached
            t0 = time.perf_counter()
            with st.spinner("Thinking..."):
                t_gen = time.perf_counter()
                sql = generate_sql(prompt)
                gen_s = time.perf_counter() - t_gen

                t_q = time.perf_counter()
                result = read(sql)               # cached unless bypass is on
                query_s = time.perf_counter() - t_q
            timing = {
                "sql_gen": gen_s,
                "query": query_s,
                "total": time.perf_counter() - t0,
                # a sub-10ms read is a cache hit, not a real DB round-trip
                "cached": (not bypass) and query_s < 0.01,
            }
            st.markdown(f"Returned {len(result)} row(s).")
            st.caption(_timing_caption(timing))
            show_result(result, sql, view_mode, key=f"live_{idx}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Returned {len(result)} row(s).",
                 "sql": sql, "table": result, "timing": timing}
            )
        except Exception as e:
            msg = f"Couldn't run that: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

# ── Voice input ───────────────────────────────────────────────────────────────
# st.audio_input records from the browser mic (requires Streamlit >= 1.36 and a
# secure context: https:// or localhost). It returns WAV bytes which we send to
# Groq Whisper. The transcript is dropped INTO the chat box (via its session_state
# key) so the user can read / edit it and press Enter to send — nothing is
# submitted automatically. We remember the last recording so each one is
# transcribed once, not on every rerun.
mic_col, tip_col = st.columns([1, 3], vertical_alignment="center")
with mic_col:
    audio = st.audio_input("Speak your question", key="voice_input",
                           label_visibility="collapsed")
with tip_col:
    st.caption("🎤 Record a question — the text lands in the box below to review "
               "and edit, then press Enter to send. Or just type.")

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_id = hash(audio_bytes)                      # identify this recording
    if st.session_state.get("last_audio_id") != audio_id:   # new recording only
        st.session_state["last_audio_id"] = audio_id  # mark as handled
        try:
            with st.spinner("Transcribing..."):
                text = transcribe_audio(audio_bytes)
            if text:
                # Prefill the chat box. This MUST run before st.chat_input below
                # is instantiated; the box then shows the text for the user to
                # verify/edit. chat_input only returns it once the user submits.
                st.session_state["chat_box"] = text
            else:
                st.warning("Didn't catch anything — please try recording again.")
        except Exception as e:
            st.error(f"Couldn't transcribe audio: {e}")

# Typed or voice-prefilled: the user reviews/edits here, then submits with Enter.
# chat_input must stay at the top level (not nested in columns/containers).
if prompt := st.chat_input(
    "e.g. Net sales by store for the period 01/07/2025 to 31/07/2025",
    key="chat_box",
):
    handle_prompt(prompt)
