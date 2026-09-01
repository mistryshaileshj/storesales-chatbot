"""
Retail Chain Store Sales Chatbot — natural language -> T-SQL -> result.
Data source: Microsoft SQL Server (live query) via SQLAlchemy + pyodbc.
Question understanding: Groq LLM (nothing hard-coded).

This follows the design and inner workings of `retailsales_chatbot.py`: a
free-form question is sent to a Groq model together with a SEMANTIC LAYER that
fixes the retail business meaning of every column and metric; the model returns
ONE T-SQL SELECT, which is guard-railed and executed read-only against SQL
Server, then rendered as a table (with hierarchical sub-totals / grand totals)
and/or a chart.

The semantic layer additionally teaches the MARGIN definitions from
`retailsales_dashboard.py`, so margin questions are answered with the provided
formula:

    cost            = SUM(purrate | Sales) - SUM(purrate | Return)
    net_taxable     = SUM(taxable_amt)
    margin          = net_taxable - cost
    profit margin % = (net_taxable - cost) * 100 / net_taxable

Carried over from the reference chatbot:
  * SQL Server connection (cached engine, read-only queries, Decimal -> numeric
    fix-up, column introspection so the model only references real columns).
  * Voice input — record a question with the mic; transcribed by Groq Whisper
    and dropped into the chat box to review/edit before sending.
  * Hierarchical sub-totals + grand total in the "Table only" view.
  * Whole-number formatting on measures (2 decimals on percentages, € on money),
    proper-case labels, decimal-free chart data labels.
  * Generated SQL kept in an expander for transparency/debugging.

Setup:
    pip install -r requirements.txt
    # requires a SQL Server ODBC driver on the host, e.g.
    #   "ODBC Driver 17 for SQL Server" or "ODBC Driver 18 for SQL Server"
    # put your secrets in .streamlit/secrets.toml:
    #   GROQ_API_KEY   = "gsk_..."
    #   RT_DB_SERVER   = "..."
    #   RT_DB_PORT     = 1433
    #   RT_DB_DATABASE = "..."
    #   RT_DB_USERNAME = "..."
    #   RT_DB_PASSWORD = "..."
    #   RT_DB_DRIVER   = "ODBC Driver 18 for SQL Server"
    streamlit run retail_sales_chatbot.py

Note: the mic needs Streamlit >= 1.36 and a secure context (https:// or
localhost).
"""

import re

import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from groq import Groq

def _resolve_table() -> str:
    """Name of the SQL Server table/view that holds the retail sales rows.
    Prefer RT_DB_TABLE from secrets so it can be changed WITHOUT editing code;
    otherwise fall back to the default below. It must expose the columns
    referenced in the semantic layer (Store, state_name, Ctr_Name, PDoc_No,
    DOC_DT, Trj_Type, qty, rtn_qty, NetItmAmt, taxable_amt, purrate, ...)."""
    try:
        return st.secrets.get("HS_DB_TABLE", "DailySalesSummary_dash")
    except Exception:
        return "DailySalesSummary_dash"


TABLE = _resolve_table()
GROQ_MODEL = "openai/gpt-oss-120b"   # VERIFY current IDs at console.groq.com/docs/models
# Speech-to-text (voice input). Groq's ASR API is OpenAI-compatible; turbo is the
# fastest/cheapest Whisper. Alternatives: "whisper-large-v3", "distil-whisper-large-v3-en".
GROQ_STT_MODEL = "whisper-large-v3-turbo"

# ---------------------------------------------------------------------------
# Database connection (Microsoft SQL Server via SQLAlchemy + pyodbc)
# Credentials live in .streamlit/secrets.toml.
# ---------------------------------------------------------------------------
def _db_config() -> dict:
    return {
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
    """
    cfg = _db_config()
    url = URL.create(
        "mssql+pyodbc",
        username=cfg["username"],
        password=cfg["password"],          # URL.create escapes special chars safely
        host=cfg["server"],
        port=cfg["port"],
        database=cfg["database"],
        query={
            "driver": cfg["driver"],
            "TrustServerCertificate": "yes",   # needed for most internal/self-signed servers
            "Encrypt": "no",
        },
    )
    return create_engine(url, pool_pre_ping=True)


def run_query(sql: str) -> pd.DataFrame:
    """Execute a read-only SELECT and return a DataFrame.
    exec_driver_sql sends the string straight to the DBAPI, so colons in
    literals aren't mistaken for bind parameters."""
    with get_engine().connect() as conn:
        res = conn.exec_driver_sql(sql)
        rows = res.fetchall()
        cols = list(res.keys())
    df = pd.DataFrame.from_records(rows, columns=cols)
    # pyodbc returns SQL Server decimal/money/numeric as Python Decimal objects,
    # which land in pandas as `object` dtype and are NOT seen as numeric — so the
    # chart logic mistakes a measure (e.g. SUM(NetItmAmt)) for a dimension.
    # Convert any object column that is fully numeric back to real numbers.
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().any() and conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_table_columns() -> list[str]:
    """Introspect the actual columns of the table/view so the model only ever
    references real column names."""
    sql = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        f"WHERE TABLE_NAME = '{TABLE}' ORDER BY ORDINAL_POSITION"
    )
    df = run_query(sql)
    return df["COLUMN_NAME"].tolist() if not df.empty else []


# ---------------------------------------------------------------------------
# Semantic layer  (retail meaning + margin formula, expressed as T-SQL)
# ---------------------------------------------------------------------------
SEMANTIC_LAYER_TMPL = """
Table `{table}` — one row per line item of a retail transaction.

Key columns (fixed business meaning):
- DOC_DT        : business/transaction date. USE THIS for any period, "per day",
                  "monthly", "over time", or date-range filter. Time part is
                  always 00:00:00, so CAST(DOC_DT AS DATE) is the day key.
- Created_Dt    : system clock timestamp. Do NOT use for business-day analysis;
                  only for "what time of day" questions.
- Trj_Type      : 'Sales' (a sale) or a RETURN. A return row has
                  Trj_Type IN ('Sales Rtn', 'Sales Return'). CRITICAL: always
                  treat BOTH 'Sales Rtn' and 'Sales Return' as returns.
- PDoc_No       : bill / transaction number. One bill spans many line rows, so a
                  COUNT of transactions/bills/orders must be
                  COUNT(DISTINCT PDoc_No), never COUNT(*).
- qty           : sold quantity. Positive on 'Sales' rows, 0 on return rows.
- rtn_qty       : returned quantity, POSITIVE on return rows, 0 on 'Sales' rows.
- NetItmAmt     : line net amount. Treat as the revenue measure unless the user
                  names a different one.
- taxable_amt   : net taxable value (already the net taxable figure).
- purrate       : the COST VALUE ITSELF for the line (NOT a per-unit rate; do NOT
                  multiply by quantity).
- Tax_Amt, DRS_amount, tottrans, TotQty : other money/quantity columns; use only if asked.
- Store         : the STORE (use for "store" / "outlet" / "branch" / "shop").
- state_name    : the STATE / region (use for "state").
- Ctr_Name      : the COUNTER / till within a store (use for "counter" / "till").
- Login_Name    : who processed the sale.

The full, authoritative column list for `{table}` is:
{columns}
Use those exact names. NEVER invent a column that is not in that list.

METRIC CONVENTIONS (follow exactly; these are different queries):
- "quantity" / "qty" / "sold" / "selling"  -> SUM(qty) WHERE Trj_Type = 'Sales'
- "net quantity" / "net qty"  (units sold minus units returned)
      -> SUM(CASE WHEN Trj_Type = 'Sales' THEN qty ELSE 0 END)
         - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN rtn_qty ELSE 0 END)
- "net sales" / "net sales value" / "net revenue"  (sales value minus returns value)
      -> SUM(CASE WHEN Trj_Type = 'Sales' THEN NetItmAmt ELSE 0 END)
         - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN NetItmAmt ELSE 0 END)
- "revenue" / "sales value"  -> SUM(NetItmAmt) WHERE Trj_Type = 'Sales'
- "returns"                  -> SUM(rtn_qty) WHERE Trj_Type IN ('Sales Rtn','Sales Return')
- "transactions" / "total transactions" / "no. of transactions" / "transaction
  count" / "bills" / "bill count" / "total bills" / "orders" / "invoices" /
  "how many" / "count of" (bills/transactions/orders/invoices/footfall)
      -> COUNT(DISTINCT PDoc_No)
  Do NOT use the `tottrans` column for a transaction/bill count — `tottrans` is
  a stored line attribute, NOT the number of bills. The bill count is ALWAYS
  COUNT(DISTINCT PDoc_No).
- "average sales" / "average sales value" / "average bill"  (per TRANSACTION)
      -> SUM(NetItmAmt) / COUNT(DISTINCT PDoc_No)
         NOT AVG(NetItmAmt).

MARGIN (use these EXACT definitions — carried over from the dashboard):
- cost         -> SUM(CASE WHEN Trj_Type = 'Sales' THEN purrate ELSE 0 END)
                  - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN purrate ELSE 0 END)
- net_taxable  -> SUM(taxable_amt)
- "margin" / "margin amount" / "margin value"  (an absolute money amount)
      -> SUM(taxable_amt)
         - ( SUM(CASE WHEN Trj_Type = 'Sales' THEN purrate ELSE 0 END)
             - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN purrate ELSE 0 END) )
- "profit margin" / "margin %" / "margin percentage" / "profitability"  (a percentage)
      -> ( SUM(taxable_amt)
           - ( SUM(CASE WHEN Trj_Type = 'Sales' THEN purrate ELSE 0 END)
               - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN purrate ELSE 0 END) ) )
         * 100.0 / NULLIF(SUM(taxable_amt), 0)
  Bare "margin" with no qualifier and words like "profitable"/"profit margin"
  mean the PERCENTAGE. Use "margin amount"/"margin value" for the money figure.
  Always wrap the divisor in NULLIF(..., 0) to avoid divide-by-zero.

Use SUM by default. Only average when the user says "average".

DIMENSIONS:
- "store" -> Store ; "state" -> state_name ; "counter"/"till" -> Ctr_Name.

DATES & PERIODS:
- Dates in questions are day-first: dd/MM/yyyy. So 03/04/2026 = 3 April 2026.
- "for the period X to Y" -> filter with
      CAST(DOC_DT AS DATE) BETWEEN 'yyyy-mm-dd' AND 'yyyy-mm-dd'
- A SINGLE named month such as "January 2026", "Jan 2026" or "in Jan 2026"
  means that WHOLE month. Filter it with a half-open range so month length
  never matters (recognise full and abbreviated month names):
      CAST(DOC_DT AS DATE) >= '<yyyy-mm-01>'
      AND CAST(DOC_DT AS DATE) < '<first day of the NEXT month>'
  e.g. January 2026  -> >= '2026-01-01' AND < '2026-02-01'.
- A single named year such as "2026" means that whole year:
      CAST(DOC_DT AS DATE) >= '2026-01-01' AND CAST(DOC_DT AS DATE) < '2027-01-01'.
- For "per month" / "monthly" (a breakdown OVER months), group by
  FORMAT(DOC_DT, 'yyyy-MM'). A single named month is a FILTER, not a GROUP BY.

TOP / BOTTOM:
- "top" / "best" / "highest" -> ORDER BY <measure> DESC with SELECT TOP N.
- "bottom" / "worst" / "lowest" / "least" -> ORDER BY <measure> ASC with SELECT TOP N.
- A singular "the top/bottom performing store/state" means N = 1; use the given
  number for "top 5", "bottom 3", etc.
"""

FEW_SHOTS = f"""
Q: Top performing store based on net sales
SQL: SELECT TOP 1 Store,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN NetItmAmt ELSE 0 END)
            - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN NetItmAmt ELSE 0 END) AS net_sales
     FROM {TABLE}
     GROUP BY Store ORDER BY net_sales DESC;

Q: Bottom 5 stores based on net sales for the period 08/03/2026 to 18/04/2026
SQL: SELECT TOP 5 Store,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN NetItmAmt ELSE 0 END)
            - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN NetItmAmt ELSE 0 END) AS net_sales
     FROM {TABLE}
     WHERE CAST(DOC_DT AS DATE) BETWEEN '2026-03-08' AND '2026-04-18'
     GROUP BY Store ORDER BY net_sales ASC;

Q: Top performing store based on profit margin
SQL: SELECT TOP 1 Store,
            (SUM(taxable_amt)
             - (SUM(CASE WHEN Trj_Type = 'Sales' THEN purrate ELSE 0 END)
                - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN purrate ELSE 0 END)))
            * 100.0 / NULLIF(SUM(taxable_amt), 0) AS margin_pct
     FROM {TABLE}
     GROUP BY Store ORDER BY margin_pct DESC;

Q: Top 3 stores by margin amount
SQL: SELECT TOP 3 Store,
            SUM(taxable_amt)
            - (SUM(CASE WHEN Trj_Type = 'Sales' THEN purrate ELSE 0 END)
               - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN purrate ELSE 0 END)) AS margin
     FROM {TABLE}
     GROUP BY Store ORDER BY margin DESC;

Q: Top performing store based on transactions
SQL: SELECT TOP 1 Store, COUNT(DISTINCT PDoc_No) AS txns
     FROM {TABLE}
     GROUP BY Store ORDER BY txns DESC;

Q: Total transactions for Jan 2026
SQL: SELECT COUNT(DISTINCT PDoc_No) AS total_transactions
     FROM {TABLE}
     WHERE CAST(DOC_DT AS DATE) >= '2026-01-01' AND CAST(DOC_DT AS DATE) < '2026-02-01';

Q: How many bills were there between 08/03/2026 and 18/04/2026
SQL: SELECT COUNT(DISTINCT PDoc_No) AS total_transactions
     FROM {TABLE}
     WHERE CAST(DOC_DT AS DATE) BETWEEN '2026-03-08' AND '2026-04-18';

Q: Top performing state based on net sales
SQL: SELECT TOP 1 state_name,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN NetItmAmt ELSE 0 END)
            - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN NetItmAmt ELSE 0 END) AS net_sales
     FROM {TABLE}
     GROUP BY state_name ORDER BY net_sales DESC;

Q: Bottom performing state based on profit margin
SQL: SELECT TOP 1 state_name,
            (SUM(taxable_amt)
             - (SUM(CASE WHEN Trj_Type = 'Sales' THEN purrate ELSE 0 END)
                - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN purrate ELSE 0 END)))
            * 100.0 / NULLIF(SUM(taxable_amt), 0) AS margin_pct
     FROM {TABLE}
     GROUP BY state_name ORDER BY margin_pct ASC;

Q: Top 5 states by transactions
SQL: SELECT TOP 5 state_name, COUNT(DISTINCT PDoc_No) AS txns
     FROM {TABLE}
     GROUP BY state_name ORDER BY txns DESC;

Q: Net sales per month for the period 08/03/2026 to 18/04/2026
SQL: SELECT FORMAT(DOC_DT, 'yyyy-MM') AS month,
            SUM(CASE WHEN Trj_Type = 'Sales' THEN NetItmAmt ELSE 0 END)
            - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN NetItmAmt ELSE 0 END) AS net_sales
     FROM {TABLE}
     WHERE CAST(DOC_DT AS DATE) BETWEEN '2026-03-08' AND '2026-04-18'
     GROUP BY FORMAT(DOC_DT, 'yyyy-MM') ORDER BY month;

Q: Net sales and profit margin by store for the top 5 stores by net sales
SQL: WITH top_stores AS (
       SELECT TOP 5 Store FROM {TABLE}
       GROUP BY Store
       ORDER BY SUM(CASE WHEN Trj_Type = 'Sales' THEN NetItmAmt ELSE 0 END)
                - SUM(CASE WHEN Trj_Type IN ('Sales Rtn','Sales Return') THEN NetItmAmt ELSE 0 END) DESC
     )
     SELECT s.Store,
            SUM(CASE WHEN s.Trj_Type = 'Sales' THEN s.NetItmAmt ELSE 0 END)
            - SUM(CASE WHEN s.Trj_Type IN ('Sales Rtn','Sales Return') THEN s.NetItmAmt ELSE 0 END) AS net_sales,
            (SUM(s.taxable_amt)
             - (SUM(CASE WHEN s.Trj_Type = 'Sales' THEN s.purrate ELSE 0 END)
                - SUM(CASE WHEN s.Trj_Type IN ('Sales Rtn','Sales Return') THEN s.purrate ELSE 0 END)))
            * 100.0 / NULLIF(SUM(s.taxable_amt), 0) AS margin_pct
     FROM {TABLE} s
     WHERE s.Store IN (SELECT Store FROM top_stores)
     GROUP BY s.Store ORDER BY net_sales DESC;
"""


def build_system_prompt(columns: list[str] | None = None) -> str:
    cols = columns if columns is not None else get_table_columns()
    col_block = ", ".join(cols) if cols else "(column list unavailable)"
    semantic = SEMANTIC_LAYER_TMPL.format(table=TABLE, columns=col_block)
    return f"""You translate a question about retail sales into ONE Microsoft SQL Server
(T-SQL) SELECT statement. Output ONLY the SQL — no prose, no markdown fences.

{semantic}

Examples:
{FEW_SHOTS}

Rules:
- This is T-SQL (Microsoft SQL Server). Use SELECT TOP N for "top/highest",
  NOT LIMIT. There is no LIMIT clause in SQL Server.
- Exactly one statement, must start with SELECT or WITH.
- Never INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/EXEC/MERGE/GRANT.
- Always add a sensible TOP N for "top/bottom/highest/lowest" style questions
  (TOP 1 for a singular "the top/bottom performing ...").
- Treat BOTH 'Sales Rtn' and 'Sales Return' as returns everywhere.
- For margin questions, use the EXACT margin / profit-margin % formula from the
  semantic layer, and wrap the divisor in NULLIF(SUM(taxable_amt), 0).
- When the user names two dimensions (e.g. "by store and counter"), GROUP BY
  both and SELECT both plus the measure, so the result draws as a stacked chart.
- CRITICAL for a "top N" with two dimensions: apply TOP to the PRIMARY dimension
  via a CTE (top N of dim1 by the measure), then return ALL rows of dim2 for
  those. NEVER put TOP on the (dim1, dim2) combination.
- Group by the date EXPRESSION, not its alias (T-SQL can't GROUP BY an alias):
  e.g. GROUP BY CAST(DOC_DT AS DATE), not GROUP BY day.
- Reference ONLY columns from the column list above. If the question is
  impossible with the schema, return exactly: SELECT 'unanswerable' AS note;
- Output the SQL and nothing else. Begin your reply directly with SELECT or
  WITH — no preamble, no explanation, no markdown fences.
"""


# ---------------------------------------------------------------------------
# SQL guardrail
# ---------------------------------------------------------------------------
_BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|exec|execute|merge|"
    r"grant|revoke|backup|restore|shutdown|waitfor|attach|copy|pragma|call|export)\b",
    re.IGNORECASE,
)


# Map the Unicode look-alikes that LLMs sometimes emit back to plain ASCII SQL.
_UNICODE_FIXES = {
    "\u2265": ">=", "\u2264": "<=", "\u2260": "<>",     # ≥ ≤ ≠
    "\u2013": "-", "\u2014": "-", "\u2212": "-",         # – — −  (dashes/minus)
    "\u2018": "'", "\u2019": "'", "\u201b": "'",         # ‘ ’ ‛  (smart single quotes)
    "\u201c": '"', "\u201d": '"',                          # “ ”     (smart double quotes)
    "\u00a0": " ", "\u2009": " ", "\u200b": "",          # nbsp / thin space / zero-width
    "\uff1d": "=",                                          # ＝ fullwidth equals
}


def _normalize_sql(sql: str) -> str:
    for bad, good in _UNICODE_FIXES.items():
        sql = sql.replace(bad, good)
    return sql


def validate_sql(sql: str) -> str:
    sql = _normalize_sql(sql).strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise ValueError("Query must start with SELECT or WITH.")
    if _BLOCKED.search(sql):
        raise ValueError("Query contains a forbidden keyword.")
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
    if not text or not text.strip():
        raise ValueError("The model returned an empty response. Please try rephrasing.")
    # If the reply has fenced code blocks, prefer the one that actually holds SQL.
    for block in re.findall(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        if re.search(r"\b(SELECT|WITH)\b", block, re.IGNORECASE):
            text = block
            break
    m = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if not m:
        raise ValueError("The model did not return an SQL query. Please try rephrasing.")
    return text[m.start():].split(";")[0].strip()


def _call_model(question: str, system: str) -> str:
    """One call to the model; return the SQL-bearing text.

    Reasoning models (e.g. gpt-oss) sometimes leave `content` empty and put the
    answer in `reasoning`, or vice-versa — so search BOTH, content first."""
    resp = get_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=2048,            # headroom so reasoning can't crowd out the SQL
        reasoning_effort="low",     # gpt-oss: minimize reasoning for this simple task
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    msg = resp.choices[0].message
    parts = [(msg.content or "").strip(), (getattr(msg, "reasoning", "") or "").strip()]
    return "\n".join(p for p in parts if p)


def generate_sql(question: str) -> str:
    system = build_system_prompt()
    try:
        return validate_sql(_extract_sql(_call_model(question, system)))
    except ValueError:
        # One strict retry: some replies come back without a clean SQL block.
        strict = system + ("\n\nIMPORTANT: Reply with ONLY the single T-SQL "
                           "SELECT (or WITH) statement. Start at SELECT/WITH. No "
                           "prose, no explanation, no code fences.")
        return validate_sql(_extract_sql(_call_model(question, strict)))


# ---------------------------------------------------------------------------
# Speech-to-text (voice input)
# ---------------------------------------------------------------------------
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe WAV audio bytes (from st.audio_input) to text via Groq Whisper."""
    resp = get_client().audio.transcriptions.create(
        file=("voice.wav", audio_bytes),   # (filename, bytes) — st.audio_input yields WAV
        model=GROQ_STT_MODEL,
        response_format="text",            # returns a plain string, not JSON
    )
    return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()


# ---------------------------------------------------------------------------
# Charting / formatting helpers
# ---------------------------------------------------------------------------
def _pretty(name: str) -> str:
    """Return a proper-case, human-friendly label for a column / alias."""
    return str(name).replace("_", " ").title()


# Column-name heuristics for display formatting (purely cosmetic; does not
# constrain what can be asked). Percentage wins over money when both match.
def _is_pct_col(name: str) -> bool:
    k = str(name).lower()
    return ("pct" in k or "percent" in k or k.endswith("%")
            or "margin_percentage" in k)


_MONEY_HINTS = ("sales", "revenue", "amount", "amt", "margin", "cost", "value",
                "price", "turnover", "taxable")


def _is_money_col(name: str) -> bool:
    k = str(name).lower()
    return (not _is_pct_col(k)) and any(h in k for h in _MONEY_HINTS)


def _round_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Round measures for display: percentages to 2 dp, everything else to whole
    numbers (kept as nullable Int64 so NaNs survive)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            if _is_pct_col(c):
                out[c] = out[c].round(2)
            else:
                out[c] = out[c].round(0).astype("Int64")
    return out


def _fmt_for(col: str) -> str:
    if _is_pct_col(col):   return "{:,.2f}%"
    if _is_money_col(col): return "€{:,.0f}"
    return "{:,.0f}"


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
    is_pct = _is_pct_col(measure)
    is_money = _is_money_col(measure)
    d = df.copy()
    for c in dim:
        d[c] = d[c].fillna("Unknown").astype(str)
    x = dim[0]
    order = (d.groupby(x)[measure].sum()
               .sort_values(ascending=False).index.tolist())
    labels = {c: _pretty(c) for c in d.columns}
    text_fmt = ".2f" if is_pct else ",.0f"
    if len(dim) == 1:
        fig = px.bar(d, x=x, y=measure, text_auto=text_fmt,
                     category_orders={x: order}, labels=labels)
        fig.update_traces(textposition="outside")
    else:
        color = dim[1]
        fig = px.bar(d, x=x, y=measure, color=color, barmode="stack",
                     text_auto=text_fmt, category_orders={x: order}, labels=labels)
        fig.update_layout(legend_title_text=_pretty(color))
    ytitle = _pretty(measure) + (" (%)" if is_pct else (" (€)" if is_money else ""))
    fig.update_layout(
        xaxis_title=_pretty(x), yaxis_title=ytitle,
        height=650, margin=dict(l=50, r=20, t=30, b=90),
        showlegend=True, font=dict(color="black"),
    )
    fig.update_yaxes(tickformat=".2f" if is_pct else ",.0f")
    st.plotly_chart(fig, use_container_width=True, key=key)


def _show_table(df: pd.DataFrame):
    """Display a table with a 1-based Sr. No column instead of the 0-based index.
    Measures are rounded (2 dp for %, whole otherwise) and headers proper-cased."""
    disp = _round_measures(df)
    fmt = {c: _fmt_for(c) for c in disp.columns if pd.api.types.is_numeric_dtype(df[c])}
    disp = disp.rename(columns={c: _pretty(c) for c in disp.columns})
    fmt = {_pretty(c): f for c, f in fmt.items()}
    disp.insert(0, "Sr. No", range(1, len(disp) + 1))
    st.dataframe(disp.style.format(fmt), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Totals for the "Table only" view
#   1 dimension    -> a Grand Total row.
#   2+ dimensions  -> HIERARCHICAL sub-totals at every grouping level, then a
#                     Grand Total. Only additive measures are totalled; a measure
#                     whose name implies an average / ratio / percentage is left
#                     blank instead of showing a meaningless sum.
# ---------------------------------------------------------------------------
_NON_ADDITIVE = ("avg", "average", "mean", "ratio", "per ", "pct", "percent", "%")

DETAIL_LEVEL = -1
GRAND_LEVEL = -2

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
    return {m: (block[m].sum() if _is_additive(m) else pd.NA) for m in measures}


def _build_totals(df: pd.DataFrame, dims: list, measures: list):
    d = df.copy()
    for c in dims:
        d[c] = d[c].astype(object)
    rows, is_total, total_level = [], [], []

    def _ordered_values(block, dim):
        additive = [m for m in measures if _is_additive(m)]
        if additive:
            return (block.groupby(dim)[additive[0]].sum()
                         .sort_values(ascending=False).index.tolist())
        return list(pd.unique(block[dim]))

    def emit_detail(block):
        for _, r in block.iterrows():
            rows.append(r.to_dict()); is_total.append(False)
            total_level.append(DETAIL_LEVEL)

    def recurse(block, level):
        dim = dims[level]
        is_leaf = (level == len(dims) - 1)
        for v in _ordered_values(block, dim):
            sub_block = block[block[dim] == v]
            if is_leaf:
                emit_detail(sub_block)
            else:
                recurse(sub_block, level + 1)
                sub = {c: "" for c in d.columns}
                sub[dim] = f"{v} \u2014 Total"
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
    for m in measures:
        vals = pd.to_numeric(out[m], errors="coerce")
        out[m] = vals.round(2) if _is_pct_col(m) else vals.round(0).astype("Int64")
    return out, is_total, total_level


def _show_table_with_totals(df: pd.DataFrame, dims: list, measures: list, key=None):
    aug, is_total, total_level = _build_totals(df, dims, measures)
    fmt = {c: _fmt_for(c) for c in measures}
    disp = aug.rename(columns={c: _pretty(c) for c in aug.columns})
    srno, n = [], 0
    for t in is_total:
        if t:
            srno.append("")
        else:
            n += 1
            srno.append(n)
    disp.insert(0, "Sr. No", srno)

    measure_labels = [_pretty(m) for m in measures]
    fmt = {_pretty(m): f for m, f in fmt.items()}

    def _highlight(row):
        lvl = total_level[row.name]
        if lvl == DETAIL_LEVEL:
            return [""] * len(row)
        bg = _total_shade(lvl)
        return [f"background-color: {bg}; font-weight: bold;"] * len(row)

    styler = (disp.style
                  .apply(_highlight, axis=1)
                  .format(fmt, subset=measure_labels, na_rep=""))
    st.dataframe(styler, use_container_width=True, hide_index=True, key=key)


def show_result(table: pd.DataFrame, sql: str, mode: str, key: str = None):
    with st.expander("SQL"):
        st.code(sql, language="sql")
    can = chartable(table)
    num, dim = _split_cols(table)
    # A single scalar answer (1 row, one measure, no dimension) — e.g. a total
    # transaction/bill count — reads better as a headline number than a table.
    if len(table) == 1 and len(num) == 1 and len(dim) == 0:
        col = num[0]
        val = table.iloc[0][col]
        val_txt = ("—" if pd.isna(val)
                   else _fmt_for(col).format(float(val)))
        st.metric(_pretty(col), val_txt)
        return
    if len(table) <= 1:                 # single row with a label -> table only, no graph
        _show_table(table)
        return
    if mode == "Table only":
        if len(dim) >= 1 and len(num) >= 1:
            _show_table_with_totals(table, dim, num, key=key)
        else:
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
# Page + sticky header
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Retail Store Sales Chatbot", layout="wide")
    st.markdown("""
    <style>
      div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
      div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
      div[data-testid="stChatInput"] textarea { min-height: 2.2rem; }
      div[data-testid="stChatMessage"] { padding: 0.35rem 0.6rem; }
      div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
          position: sticky; top: 0; z-index: 999;
          background-color: var(--background-color, white);
      }
    </style>
    """, unsafe_allow_html=True)

    header = st.container()
    header.markdown(
        "<h1 style='font-size:1.6rem; margin:0 0 0.3rem 0;'>🛍️ Retail Store Sales Chatbot</h1>",
        unsafe_allow_html=True,
    )
    c1, c2 = header.columns([1, 10], vertical_alignment="center")
    c1.markdown("**Display**")
    view_mode = c2.radio(
        "Display", ["Table + Graph", "Table only", "Graph only"],
        index=1, horizontal=True, label_visibility="collapsed", key="view_mode",
    )
    header.markdown('<div class="fixed-header"></div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Prompt history")
        search = st.text_input("Search", placeholder="filter prompts...")
        user_msgs = [(i, m["content"]) for i, m in enumerate(st.session_state.messages)
                     if m["role"] == "user"]
        if not user_msgs:
            st.caption("No prompts yet.")
        for i, text in reversed(user_msgs):
            if search and search.lower() not in text.lower():
                continue
            label = text if len(text) <= 45 else text[:42] + "..."
            safe = text.replace('"', "&quot;")
            st.markdown(f'<a href="#msg{i}" title="{safe}">{label}</a>',
                        unsafe_allow_html=True)

    for i, m in enumerate(st.session_state.messages):
        if m["role"] == "user":
            st.markdown(f"<div id='msg{i}'></div>", unsafe_allow_html=True)
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("table") is not None:
                show_result(m["table"], m["sql"], view_mode, key=f"hist_{i}")

    def handle_prompt(prompt: str):
        idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div id='msg{idx}'></div>", unsafe_allow_html=True)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    sql = generate_sql(prompt)
                    result = run_query(sql)
                st.markdown(f"Returned {len(result)} row(s).")
                show_result(result, sql, view_mode, key=f"live_{idx}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Returned {len(result)} row(s).",
                     "sql": sql, "table": result}
                )
            except Exception as e:
                msg = f"Couldn't run that: {e}"
                if "Invalid object name" in str(e) or "42S02" in str(e):
                    msg += (f"\n\nThe table/view `{TABLE}` doesn't exist in this "
                            "database. Set the correct name — either `RT_DB_TABLE` "
                            "in `.streamlit/secrets.toml`, or the `TABLE` constant "
                            "at the top of the app.")
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

    # ── Voice input ──────────────────────────────────────────────────────
    mic_col, tip_col = st.columns([1, 3], vertical_alignment="center")
    with mic_col:
        audio = st.audio_input("Speak your question", key="voice_input",
                               label_visibility="collapsed")
    with tip_col:
        st.caption("🎤 Record a question — the text lands in the box below to review "
                   "and edit, then press Enter to send. Or just type.")

    if audio is not None:
        audio_bytes = audio.getvalue()
        audio_id = hash(audio_bytes)
        if st.session_state.get("last_audio_id") != audio_id:
            st.session_state["last_audio_id"] = audio_id
            try:
                with st.spinner("Transcribing..."):
                    text = transcribe_audio(audio_bytes)
                if text:
                    st.session_state["chat_box"] = text
                else:
                    st.warning("Didn't catch anything — please try recording again.")
            except Exception as e:
                st.error(f"Couldn't transcribe audio: {e}")

    if prompt := st.chat_input(
        "e.g. Top performing store for the period 08/03/2026 to 18/04/2026 based on net sales",
        key="chat_box",
    ):
        handle_prompt(prompt)


if __name__ == "__main__":
    main()
