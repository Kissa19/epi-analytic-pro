import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import timedelta, datetime, date
import scipy.stats as stats
from scipy.stats import hypergeom, chi2_contingency
import plotly.express as px
import folium
from streamlit_folium import folium_static
import requests
import math
import re
import io
import hashlib
import platform
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor

APP_NAME = "ระบบวิเคราะห์ข้อมูลระบาดวิทยาขั้นสูงแบบรวมศูนย์ (Epi-Analytic Pro)"
APP_VERSION = "Research build HE69-085 v1.0"
PII_COLUMN_PATTERNS = [
    r"(^|[_\s-])(ชื่อ|นามสกุล|name|surname|firstname|lastname)([_\s-]|$)",
    r"เลข.*บัตร|บัตร.*ประชาชน|citizen|national.?id|id.?card",
    r"(^|[_\s-])(hn|an)([_\s-]|$)",
    r"โทร|เบอร์|phone|mobile|tel",
    r"ที่อยู่|บ้านเลขที่|address",
    r"รหัส.*ผู้ป่วย|patient.?id|case.?id|person.?id|เลขที่.*ผู้ป่วย",
]

# ==========================================
# 1. CONFIGURATION & STYLING (MODERN SARABUN)
# ==========================================
st.set_page_config(
    page_title="Epi-Analytic Pro | ระบบวิเคราะห์ข้อมูลระบาดวิทยาขั้นสูงแบบรวมศูนย์", 
    page_icon="🦠", 
    layout="wide"
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700;800&family=Sarabun:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #F6F8FF;
            --surface: rgba(255,255,255,0.88);
            --surface-solid: #FFFFFF;
            --ink: #172033;
            --muted: #64748B;
            --primary: #6556FF;
            --primary-2: #00B4D8;
            --accent: #EC4899;
            --success: #10B981;
            --warning: #F59E0B;
            --border: rgba(100,116,139,0.16);
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
            --shadow-soft: 0 10px 30px rgba(15, 23, 42, 0.07);
            --radius: 22px;
        }

        html, body, [class*="css"], [class*="st-"], div, span, label, p, a, button, input, select, textarea,
        h1, h2, h3, h4, h5, h6, th, td {
            font-family: 'Noto Sans Thai', 'Sarabun', sans-serif !important;
            letter-spacing: -0.01em;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(101, 86, 255, 0.20), transparent 32rem),
                radial-gradient(circle at top right, rgba(0, 180, 216, 0.18), transparent 30rem),
                linear-gradient(180deg, #F8FAFF 0%, #F5F7FB 45%, #FFFFFF 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 3rem !important;
            max-width: 1480px !important;
        }

        h1, h2, h3 { color: var(--ink) !important; }
        h1 { font-size: clamp(2.0rem, 4vw, 3.35rem) !important; font-weight: 800 !important; letter-spacing: -0.05em; }
        h2 { font-size: 2.0rem !important; font-weight: 750 !important; }
        h3 { font-size: 1.35rem !important; font-weight: 700 !important; }
        p, span, label, div, th, td { font-size: 1.02rem !important; }

        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.86) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--border);
            box-shadow: 12px 0 40px rgba(15, 23, 42, 0.06);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label { color: var(--muted) !important; }

        .app-brand {
            padding: 18px 16px;
            border-radius: var(--radius);
            background: linear-gradient(135deg, rgba(101,86,255,0.12), rgba(0,180,216,0.12));
            border: 1px solid rgba(101,86,255,0.18);
            box-shadow: var(--shadow-soft);
            margin-bottom: 14px;
        }
        .app-brand-title { font-size: 1.25rem !important; font-weight: 800; color: var(--ink); margin-bottom: 2px; }
        .app-brand-subtitle { font-size: 0.92rem !important; color: var(--muted); }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 28px 30px;
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(18,24,38,0.96), rgba(58,45,160,0.92)),
                radial-gradient(circle at 80% 20%, rgba(0,180,216,0.36), transparent 26rem);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: var(--shadow);
            margin: 0 0 1.2rem 0;
        }
        .hero:after {
            content: "";
            position: absolute;
            right: -70px; top: -80px;
            width: 260px; height: 260px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(236,72,153,0.45), transparent 65%);
        }
        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.15);
            color: #C7D2FE;
            font-weight: 700;
            font-size: 0.86rem !important;
            margin-bottom: 12px;
        }
        .hero-title { color: #FFFFFF !important; font-size: clamp(2rem, 5vw, 3.5rem) !important; line-height: 1.05; font-weight: 800; margin: 0 0 10px 0; letter-spacing: -0.05em; }
        .hero-subtitle { color: rgba(255,255,255,0.78); max-width: 880px; font-size: 1.05rem !important; margin-bottom: 0; }

        .section-title {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 1.4rem 0 0.8rem 0;
        }
        .section-icon {
            display: grid;
            place-items: center;
            width: 46px; height: 46px;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            color: #fff;
            box-shadow: 0 10px 22px rgba(101,86,255,0.25);
            font-size: 1.25rem !important;
        }
        .section-heading { font-size: 1.75rem !important; font-weight: 800; color: var(--ink); }
        .section-caption { color: var(--muted); font-size: 0.96rem !important; margin-top: -6px; }

        .metric-card, .glass-card, .template-box, .ai-summary-box {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            box-shadow: var(--shadow-soft) !important;
        }
        .metric-card {
            padding: 16px 18px;
            min-height: 112px;
        }
        .metric-label { color: var(--muted); font-size: 0.88rem !important; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
        .metric-value { color: var(--ink); font-size: 2.0rem !important; font-weight: 800; line-height: 1.1; margin-top: 8px; }
        .metric-note { color: var(--muted); font-size: 0.88rem !important; margin-top: 4px; }

        div[data-testid="stMetric"] {
            background: var(--surface) !important;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px 18px;
            box-shadow: var(--shadow-soft);
        }
        [data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 800 !important; }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 700 !important; }

        .stButton > button, .stDownloadButton > button, button[kind="primary"] {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%) !important;
            color: #FFFFFF !important;
            border: 0 !important;
            border-radius: 16px !important;
            padding: 0.72rem 1rem !important;
            font-weight: 800 !important;
            box-shadow: 0 14px 24px rgba(101,86,255,0.22);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 30px rgba(101,86,255,0.30);
        }

        [data-baseweb="select"] > div,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stFileUploader"] section,
        textarea {
            border-radius: 16px !important;
            border-color: rgba(100,116,139,0.25) !important;
            background-color: rgba(255,255,255,0.92) !important;
        }
        div[data-testid="stDataFrame"], .stTable, [data-testid="stTable"] {
            border-radius: var(--radius) !important;
            overflow: hidden !important;
            box-shadow: var(--shadow-soft);
        }
        .template-box { padding: 18px; margin-bottom: 12px; }
        .template-link {
            color: var(--primary) !important;
            text-decoration: none;
            font-weight: 700;
            display: block;
            padding: 9px 12px;
            border-radius: 12px;
        }
        .template-link:hover { background: rgba(101,86,255,0.08); }
        .ai-summary-box {
            border-left: 6px solid var(--success) !important;
            padding: 18px 20px;
            margin-top: 15px;
            line-height: 1.75;
        }
        .small-muted { color: var(--muted); font-size: 0.92rem !important; }
        hr { border-color: rgba(100,116,139,0.16) !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. SESSION STATE & PRIVACY HELPERS
# ==========================================
for key, default in {
    'deidentified_confirmed': False,
    'research_results': {},
    'session_nonce': hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:12],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def detect_pii_columns(dataframe):
    findings = []
    for col in dataframe.columns:
        normalized = re.sub(r"\s+", " ", str(col).strip().lower())
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in PII_COLUMN_PATTERNS):
            findings.append(str(col))
    return findings

def detect_pii_values(dataframe):
    """Conservative scan of string values for Thai ID (checksum), phone, and house address."""
    findings = []
    thai_id = re.compile(r"(?<!\d)\d{13}(?!\d)")
    phone = re.compile(r"(?<!\d)(?:\+?66|0)\d{8,9}(?!\d)")
    house = re.compile(r"(?:บ้านเลขที่|เลขที่)\s*\d+(?:/\d+)?", re.IGNORECASE)
    for col in dataframe.select_dtypes(include=['object', 'string']).columns:
        sample = dataframe[col].dropna().astype(str).head(500)
        if sample.str.contains(thai_id).any() or sample.str.contains(phone).any() or sample.str.contains(house).any():
            findings.append(str(col))
    return findings

def validate_deidentified(dataframe):
    risky = sorted(set(detect_pii_columns(dataframe) + detect_pii_values(dataframe)))
    return len(risky) == 0, risky

def jitter_coordinates(dataframe, lat_col, lon_col, meters=30, seed=None):
    """Create session-only masked coordinates; original coordinates are never exported."""
    out = dataframe.copy()
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, len(out))
    distance = rng.uniform(5, max(5, meters), len(out))
    lat = pd.to_numeric(out[lat_col], errors='coerce')
    lon = pd.to_numeric(out[lon_col], errors='coerce')
    out['_masked_lat'] = lat + (distance * np.cos(angle)) / 111_320
    cos_lat = np.cos(np.radians(lat)).clip(0.1)
    out['_masked_lon'] = lon + (distance * np.sin(angle)) / (111_320 * cos_lat)
    return out

def safe_export(dataframe):
    """Export aggregated/non-identifying results only."""
    blocked = set(detect_pii_columns(dataframe))
    blocked.update(c for c in dataframe.columns if any(k in str(c).lower() for k in ['lat', 'lon', 'ละติจูด', 'ลองจิจูด']))
    return dataframe.drop(columns=list(blocked), errors='ignore')

def tables_to_excel_bytes(tables):
    """Create an in-memory Excel workbook containing aggregated tables only."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_export(table).to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
    return buffer.getvalue()

def frequency_table(series, variable_name, denominator=None, include_missing=True):
    """Return n and percent for a categorical variable without row-level data."""
    values = series.copy()
    if include_missing:
        values = values.astype("object").where(values.notna(), "ไม่ระบุ")
    else:
        values = values.dropna()
    counts = values.astype(str).value_counts(dropna=False).rename_axis("กลุ่ม").reset_index(name="n")
    denom = int(denominator if denominator is not None else counts["n"].sum())
    counts["%"] = np.where(denom > 0, counts["n"] / denom * 100, 0)
    counts.insert(0, "ตัวแปร", variable_name)
    return counts

def clear_analysis_state():
    preserve = {'session_nonce'}
    for key in list(st.session_state.keys()):
        if key not in preserve:
            del st.session_state[key]
    st.cache_data.clear()
    st.session_state['deidentified_confirmed'] = False
    st.session_state['research_results'] = {}

def icc_absolute_agreement(data):
    """ICC(A,1): two-way random effects, absolute agreement, single measurement."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        return np.nan
    n, k = x.shape
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_error = np.sum((x - row_means[:, None] - col_means[None, :] + grand) ** 2)
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return (ms_rows - ms_error) / denominator if denominator else np.nan

def make_synthetic_scenarios():
    rng = np.random.default_rng(69085)
    scenarios = []
    for name, n, effect, missing in [
        ("cohort_balanced", 120, 2.0, 0.00),
        ("case_control_confounding", 200, 2.5, 0.05),
        ("zero_cell_small_sample", 40, 8.0, 0.00),
    ]:
        exposure = rng.binomial(1, .45, n)
        confounder = rng.binomial(1, .35, n)
        logits = -1.7 + np.log(effect) * exposure + .8 * confounder
        outcome = rng.binomial(1, 1 / (1 + np.exp(-logits)))
        if "zero_cell" in name:
            outcome[(exposure == 0) & (outcome == 1)] = 0
        frame = pd.DataFrame({"scenario": name, "outcome": outcome, "exposure": exposure,
                              "confounder": confounder, "age": rng.integers(18, 80, n)})
        if missing:
            frame.loc[rng.choice(n, int(n * missing), replace=False), "confounder"] = np.nan
        scenarios.append(frame)
    return pd.concat(scenarios, ignore_index=True)

high_res_config = {
    'displaylogo': False,
    'toImageButtonOptions': {'format': 'png', 'filename': 'Epi_Chart_Export', 'height': 720, 'width': 1280, 'scale': 2}
}


def render_hero(title, subtitle, kicker="PRIVACY BY DESIGN • ODPC8"):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-kicker">✨ {kicker}</div>
        <div class="hero-title">{title}</div>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def section_header(icon, title, caption=None):
    cap_html = f'<div class="section-caption">{caption}</div>' if caption else ''
    st.markdown(f"""
    <div class="section-title">
        <div class="section-icon">{icon}</div>
        <div>
            <div class="section-heading">{title}</div>
            {cap_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def render_data_overview(dataframe):
    numeric_cols = dataframe.select_dtypes(include=[np.number]).shape[1]
    missing_pct = dataframe.isna().mean().mean() * 100 if len(dataframe) else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Rows", f"{len(dataframe):,}", "จำนวนระเบียน")
    with c2: metric_card("Columns", f"{dataframe.shape[1]:,}", "จำนวนตัวแปร")
    with c3: metric_card("Numeric", f"{numeric_cols:,}", "ตัวแปรเชิงปริมาณ")
    with c4: metric_card("Missing", f"{missing_pct:.1f}%", "ค่า missing เฉลี่ย")


def _infer_likely_column(dataframe, keywords, min_parse_ratio=0.35):
    """Find a likely column by name first, then by parse success rate."""
    for col in dataframe.columns:
        col_text = str(col).lower()
        if any(str(k).lower() in col_text for k in keywords):
            return col
    return None


def _infer_date_column(dataframe):
    """Infer onset/date column for dashboard, avoiding running-number columns."""
    name_hit = _infer_likely_column(dataframe, [
        "วันเริ่มป่วย", "เริ่มป่วย", "onset", "date_onset", "date onset", "วันที่ป่วย", "วันที่"
    ])
    if name_hit is not None:
        return name_hit

    best_col, best_ratio = None, 0
    for col in dataframe.columns:
        # Skip obvious ID/running number fields.
        col_text = str(col).lower()
        if any(k in col_text for k in ["ลำดับ", "id", "no", "number", "เลขที่"]):
            continue
        sample = dataframe[col].dropna().head(60)
        if sample.empty:
            continue
        parsed = sample.apply(parse_epi_date_value)
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio
    return best_col if best_ratio >= 0.35 else None


def render_dataset_dashboard(dataframe):
    """Modern first-look dashboard after uploading data."""
    section_header("🏠", "Dashboard สรุปข้อมูลหลังนำเข้า", "ภาพรวมคุณภาพข้อมูลและตัวแปรสำคัญก่อนเลือกวิเคราะห์เชิงลึก")
    render_data_overview(dataframe)

    total_n = len(dataframe)
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    missing_by_col = (dataframe.isna().mean() * 100).sort_values(ascending=False)
    complete_rows = int(dataframe.dropna(how="any").shape[0])

    sex_c = find_col(dataframe, ['sex', 'gender', 'เพศ'])
    age_c = find_col(dataframe, ['age', 'อายุ'])
    date_c = _infer_date_column(dataframe)
    lat_c = next((c for c in dataframe.columns if any(p in str(c).lower() for p in ['lat', 'latitude', 'ละติจูด'])), None)
    lon_c = next((c for c in dataframe.columns if any(p in str(c).lower() for p in ['lon', 'longitude', 'ลองจิจูด'])), None)

    parsed_dates = None
    if date_c:
        parsed_dates = parse_epi_date_series(dataframe[date_c])
        valid_dates = parsed_dates.dropna()
    else:
        valid_dates = pd.Series(dtype="datetime64[ns]")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Complete rows", f"{complete_rows:,}", "แถวที่ไม่มี missing เลย")
    with c2:
        if not valid_dates.empty:
            metric_card("Onset range", f"{valid_dates.min():%d/%m/%Y} - {valid_dates.max():%d/%m/%Y}", f"จากคอลัมน์ {date_c}")
        else:
            metric_card("Onset range", "ยังไม่พบ", "ยังไม่พบคอลัมน์วันที่ที่อ่านได้")
    with c3:
        if age_c:
            age_num = pd.to_numeric(dataframe[age_c], errors='coerce').dropna()
            metric_card("Age median", f"{age_num.median():.1f}" if not age_num.empty else "N/A", f"จากคอลัมน์ {age_c}")
        else:
            metric_card("Age median", "ยังไม่พบ", "ยังไม่พบคอลัมน์อายุ")
    with c4:
        geo_ready = int(dataframe[[lat_c, lon_c]].dropna().shape[0]) if lat_c and lon_c else 0
        metric_card("GIS ready", f"{geo_ready:,}", "ระเบียนที่มี Lat/Lon")

    st.markdown("---")
    left, right = st.columns([1.15, 0.85])

    with left:
        section_header("📈", "สัญญาณข้อมูลสำคัญ", "กราฟย่อสำหรับประเมินข้อมูลก่อนวิเคราะห์")
        if not valid_dates.empty:
            daily = valid_dates.dt.floor('D').value_counts().sort_index().reset_index()
            daily.columns = ["วันที่เริ่มป่วย", "จำนวน"]
            fig_daily = px.bar(daily, x="วันที่เริ่มป่วย", y="จำนวน", text_auto=True,
                               title="Mini Epidemic Curve: จำนวนผู้ป่วยตามวันเริ่มป่วย",
                               color_discrete_sequence=['#6556FF'])
            fig_daily.update_layout(font=dict(family="Sarabun", size=15), xaxis_title="วันที่เริ่มป่วย", yaxis_title="จำนวนผู้ป่วย", bargap=0.05)
            st.plotly_chart(fig_daily, use_container_width=True, config=high_res_config)
        else:
            st.info("ยังไม่พบคอลัมน์วันที่ที่ระบบอ่านได้ จึงยังไม่แสดง Mini Epi Curve")

        if age_c:
            age_num = pd.to_numeric(dataframe[age_c], errors='coerce')
            age_grp = pd.cut(age_num, bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'], right=False)
            age_df = age_grp.value_counts().sort_index().reset_index()
            age_df.columns = ["กลุ่มอายุ", "จำนวน"]
            fig_age = px.bar(age_df, x="กลุ่มอายุ", y="จำนวน", text_auto=True,
                             title="จำนวนผู้ป่วยตามกลุ่มอายุ", color_discrete_sequence=['#00B4D8'])
            fig_age.update_layout(font=dict(family="Sarabun", size=15), xaxis_title="กลุ่มอายุ", yaxis_title="จำนวน")
            st.plotly_chart(fig_age, use_container_width=True, config=high_res_config)

    with right:
        section_header("🧭", "ตัวแปรที่ระบบตรวจพบ", "ช่วยเลือกเมนูวิเคราะห์ถัดไปได้เร็วขึ้น")
        detected = pd.DataFrame([
            {"รายการ": "คอลัมน์วันเริ่มป่วย", "ตรวจพบ": date_c or "-", "สถานะ": "พร้อม" if date_c and not valid_dates.empty else "ตรวจสอบ"},
            {"รายการ": "คอลัมน์เพศ", "ตรวจพบ": sex_c or "-", "สถานะ": "พร้อม" if sex_c else "ตรวจสอบ"},
            {"รายการ": "คอลัมน์อายุ", "ตรวจพบ": age_c or "-", "สถานะ": "พร้อม" if age_c else "ตรวจสอบ"},
            {"รายการ": "คอลัมน์พิกัด Lat/Lon", "ตรวจพบ": f"{lat_c} / {lon_c}" if lat_c and lon_c else "-", "สถานะ": "พร้อม" if lat_c and lon_c else "ตรวจสอบ"},
            {"รายการ": "ตัวแปรเชิงปริมาณ", "ตรวจพบ": f"{len(numeric_cols)} ตัวแปร", "สถานะ": "พร้อม" if numeric_cols else "ตรวจสอบ"},
        ])
        st.dataframe(detected, use_container_width=True, hide_index=True)

        missing_top = missing_by_col.head(10).reset_index()
        missing_top.columns = ["ตัวแปร", "% missing"]
        fig_miss = px.bar(missing_top.sort_values("% missing"), x="% missing", y="ตัวแปร", orientation='h',
                          title="Top missing data", color_discrete_sequence=['#EC4899'])
        fig_miss.update_layout(font=dict(family="Sarabun", size=14), xaxis_title="% missing", yaxis_title="")
        st.plotly_chart(fig_miss, use_container_width=True, config=high_res_config)

        if sex_c:
            sex_df = dataframe[sex_c].astype(str).str.strip().replace({'1':'ชาย','2':'หญิง','1.0':'ชาย','2.0':'หญิง'}).value_counts().reset_index()
            sex_df.columns = ["เพศ", "จำนวน"]
            fig_sex = px.pie(sex_df, names="เพศ", values="จำนวน", hole=0.55, title="สัดส่วนตามเพศ")
            fig_sex.update_layout(font=dict(family="Sarabun", size=14))
            st.plotly_chart(fig_sex, use_container_width=True, config=high_res_config)

    st.markdown("---")
    section_header("🚀", "เมนูวิเคราะห์แนะนำ", "เลือกเมนูจากแถบซ้ายตามเป้าหมายการวิเคราะห์")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.info("📊 **Epi Curve**\n\nใช้เมื่อมีคอลัมน์วันเริ่มป่วย/เวลาเริ่มป่วย")
    with q2:
        st.info("👤 **Person**\n\nสรุปเพศ อายุ อาการ และปัจจัยอื่น ๆ")
    with q3:
        st.info("🗺️ **Place**\n\nSpot Map และตารางจำแนกตามสถานที่")
    with q4:
        st.info("🔬 **Bivariate**\n\nวิเคราะห์ OR/RR จากไฟล์ หรือกรอก Manual 2x2 ได้ทันที")

    st.info("🔒 ระบบไม่แสดงตัวอย่างข้อมูลรายแถว เพื่อลดความเสี่ยงการเปิดเผยข้อมูลรายบุคคล")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            try: return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding='cp874')
        else: return pd.read_excel(file)
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์ได้: {e}")
        return None

def smart_map_variable(series):
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({1, 2, 1.0, 2.0, '1', '2'}):
        return pd.to_numeric(series, errors='coerce').map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    return series

def calculate_mid_p(a, b, c, d):
    n = a + b + c + d
    if n == 0: return 1.0
    k, m = a + c, a + b
    p_obs = hypergeom.pmf(a, n, k, m)
    p_lower = hypergeom.cdf(a, n, k, m)
    p_upper = hypergeom.sf(a-1, n, k, m)
    mid_p = 2 * (min(p_lower, p_upper) - 0.5 * p_obs)
    return max(min(mid_p, 1.0), 0.0)

def calculate_attack_rate(cases, population):
    if population <= 0 or cases < 0 or cases > population:
        raise ValueError("cases/population ไม่ถูกต้อง")
    return cases / population * 100

def calculate_2x2(a, b, c, d, design="OR", correction=True):
    cells = np.asarray([a, b, c, d], dtype=float)
    if np.any(cells < 0):
        raise ValueError("จำนวนในตาราง 2x2 ต้องไม่ติดลบ")
    corrected = bool(np.any(cells == 0) and correction)
    aa, bb, cc, dd = cells + 0.5 if corrected else cells
    if design.upper() == "OR":
        estimate = (aa * dd) / (bb * cc)
        se = math.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
    elif design.upper() == "RR":
        estimate = (aa/(aa+bb)) / (cc/(cc+dd))
        se = math.sqrt((1/aa - 1/(aa+bb)) + (1/cc - 1/(cc+dd)))
    else:
        raise ValueError("design ต้องเป็น OR หรือ RR")
    return {"estimate": estimate,
            "lower": math.exp(math.log(estimate) - 1.96 * se),
            "upper": math.exp(math.log(estimate) + 1.96 * se),
            "mid_p": calculate_mid_p(int(a), int(b), int(c), int(d)),
            "corrected": corrected}

def find_col(df, possible_names):
    return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)


def _thai_digit_to_arabic(text):
    """Convert Thai numerals to Arabic numerals for date parsing."""
    if text is None:
        return text
    return str(text).translate(str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789'))


def _safe_datetime(year, month, day, hour=0, minute=0, second=0):
    """Create datetime and convert Buddhist Era year to Common Era when needed."""
    try:
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour or 0)
        minute = int(minute or 0)
        second = int(second or 0)
        if year >= 2400:
            year -= 543
        if 1900 <= year <= 2200:
            return datetime(year, month, day, hour, minute, second)
    except Exception:
        return pd.NaT
    return pd.NaT


def parse_epi_date_value(value):
    """Robust date parser for epidemiology datasets.

    Key rule:
    - Numeric 1, 2, 3... must NOT be treated as dates because many Thai field
      investigation sheets have a running-number column named ลำดับ.
    - Buddhist Era years such as 2568 are converted to Common Era 2025.
    """
    if pd.isna(value):
        return pd.NaT

    # Excel/openpyxl can return Python datetime or pandas Timestamp.
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return _safe_datetime(
            value.year,
            value.month,
            value.day,
            getattr(value, "hour", 0),
            getattr(value, "minute", 0),
            getattr(value, "second", 0),
        )

    # Numeric Excel serial dates:
    # - CE dates are usually around 30,000-80,000.
    # - BE serial dates can be >200,000, e.g. year 2568 in Excel-like serial.
    # - Values such as 1,2,3... are almost always ID/running numbers, not onset dates.
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        try:
            num = float(value)
            if (30000 <= num <= 80000) or (200000 <= num <= 400000):
                whole_days = math.floor(num)
                frac_seconds = round((num - whole_days) * 86400)
                dt = datetime(1899, 12, 30) + timedelta(days=whole_days, seconds=frac_seconds)
                return _safe_datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            return pd.NaT
        except Exception:
            return pd.NaT

    raw = _thai_digit_to_arabic(value).strip()
    if raw == "" or raw.lower() in {"nan", "nat", "none", "null"}:
        return pd.NaT

    raw = re.sub(r"\s*น\.?$", "", raw)
    raw = raw.replace("พ.ศ.", "").replace("พศ.", "").replace("ค.ศ.", "").replace("คศ.", "")
    raw = re.sub(r"\s+", " ", raw).strip()

    # Direct split formats:
    # yyyy-mm-dd, yyyy/mm/dd, dd/mm/yyyy, dd-mm-yyyy, with optional hh:mm:ss.
    m = re.match(r"^(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$", raw)
    if m:
        a, b, c, hh, mm, ss = m.groups()
        a_i, b_i, c_i = int(a), int(b), int(c)

        # Year-first format, including Buddhist Era: 2568-06-07.
        if a_i >= 1900:
            return _safe_datetime(a_i, b_i, c_i, hh, mm, ss)

        # Day-first format: 7/6/2568 or 07/06/2025.
        if c_i < 100:
            # Interpret 2-digit years in a public-health field dataset.
            c_i += 2500 if c_i < 80 else 2400
        return _safe_datetime(c_i, b_i, a_i, hh, mm, ss)

    # Compact formats only when there are no separators, e.g. 25680607 or 07062568.
    if re.fullmatch(r"\d{6}|\d{8}", raw):
        digits = raw
        candidates = []
        if len(digits) == 8:
            candidates.extend([
                (digits[0:4], digits[4:6], digits[6:8]),  # yyyymmdd
                (digits[4:8], digits[2:4], digits[0:2]),  # ddmmyyyy
            ])
        elif len(digits) == 6:
            yy1 = int(digits[0:2])
            yy2 = int(digits[4:6])
            candidates.extend([
                (2500 + yy1 if yy1 < 80 else 2400 + yy1, digits[2:4], digits[4:6]),  # yymmdd, BE-like
                (2500 + yy2 if yy2 < 80 else 2400 + yy2, digits[2:4], digits[0:2]),  # ddmmyy, BE-like
                (2000 + yy1 if yy1 < 80 else 1900 + yy1, digits[2:4], digits[4:6]),  # yymmdd, CE-like
                (2000 + yy2 if yy2 < 80 else 1900 + yy2, digits[2:4], digits[0:2]),  # ddmmyy, CE-like
            ])
        for y, mth, d in candidates:
            parsed = _safe_datetime(y, mth, d)
            if not pd.isna(parsed):
                return parsed

    # Last fallback for normal CE date strings. Do not use this for low integers.
    dt = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    if not pd.isna(dt):
        return _safe_datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    return pd.NaT


def parse_epi_date_series(series):
    """Parse a Streamlit-selected date column and keep valid values inside pandas bounds."""
    return series.apply(parse_epi_date_value)



def show_data_required_panel(menu_title="เมนูนี้ต้องใช้ไฟล์ข้อมูล"):
    """Render a friendly empty state when a selected analysis requires uploaded data."""
    st.markdown(
        """
        <div class="template-box" style="padding:24px; border-left:6px solid var(--primary); background:rgba(255,255,255,0.92);">
            <div style="font-size:1.35rem !important; font-weight:800; color:var(--ink); margin-bottom:8px;">📂 พร้อมวิเคราะห์ทันทีเมื่อมีข้อมูล</div>
            <div style="color:var(--muted); line-height:1.75;">
                เมนูนี้ต้องใช้ข้อมูลจำลองหรือข้อมูลที่ตัดตัวระบุบุคคลแล้วจากไฟล์ Excel/CSV
                กรุณายืนยันการทำ de-identification และอัปโหลดไฟล์จากแถบด้านซ้าย
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("1) ใช้เฉพาะ Excel/CSV ที่ตัดตัวระบุบุคคลแล้ว")
    with c2:
        st.info("2) ตรวจสอบชื่อคอลัมน์ให้ตรงกับตัวแปรที่ต้องการวิเคราะห์")
    with c3:
        st.info("3) กลับมาที่เมนูนี้ ระบบจะแสดงหน้าต่างวิเคราะห์ให้อัตโนมัติ")


def render_manual_2x2_calculator():
    """Manual 2x2 calculator that works without uploaded files."""
    st.subheader("🔢 Manual 2x2 Table Calculator")
    st.info("ใช้สำหรับคำนวณกรณีมีเพียงตัวเลขสรุป (Aggregated Data) โดยไม่ต้องอัปโหลดไฟล์")

    with st.container(border=True):
        manual_design = st.radio(
            "รูปแบบการศึกษา (Study Design):",
            ["Cohort Study (Relative Risk)", "Case-Control Study (Odds Ratio)"],
            horizontal=True, key="man_design"
        )

        st.markdown("#### ตาราง 2x2")
        st.caption("นิยาม: a = สัมผัสและป่วย, b = สัมผัสและไม่ป่วย, c = ไม่สัมผัสและป่วย, d = ไม่สัมผัสและไม่ป่วย")
        c0, c1, c2 = st.columns([1.45, 1, 1])
        with c0:
            st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
            st.markdown("**Exposed / สัมผัสปัจจัย**")
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            st.markdown("**Non-exposed / ไม่สัมผัส**")
        with c1:
            st.markdown("<center><b>Sick / ป่วย</b></center>", unsafe_allow_html=True)
            ma = st.number_input("a", min_value=0, value=0, step=1, help="Exposed + Sick")
            mc = st.number_input("c", min_value=0, value=0, step=1, help="Non-exposed + Sick")
        with c2:
            st.markdown("<center><b>Not sick / ไม่ป่วย</b></center>", unsafe_allow_html=True)
            mb = st.number_input("b", min_value=0, value=0, step=1, help="Exposed + Not sick")
            md = st.number_input("d", min_value=0, value=0, step=1, help="Non-exposed + Not sick")

        calc_col, reset_col = st.columns([2, 1])
        run_calc = calc_col.button("📈 คำนวณผล 2x2 Table", type="primary", use_container_width=True)
        if reset_col.button("🧹 ล้างผลลัพธ์", use_container_width=True):
            st.session_state.pop('biv_man_res', None)
            st.rerun()

    if run_calc:
        if (ma + mb + mc + md) > 0:
            try:
                if "Case-Control" in manual_design:
                    res_label = "Odds Ratio (OR)"
                    val = (ma * md) / (mb * mc) if (mb * mc) > 0 else 0
                    se_ln = math.sqrt(1/ma + 1/mb + 1/mc + 1/md) if ma*mb*mc*md > 0 else 0
                else:
                    res_label = "Relative Risk (RR)"
                    val = (ma / (ma + mb)) / (mc / (mc + md)) if (ma + mb) > 0 and (mc + md) > 0 else 0
                    se_ln = math.sqrt((1/ma - 1/(ma+mb)) + (1/mc - 1/(mc+md))) if ma*mc > 0 else 0

                lower = math.exp(math.log(val) - 1.96 * se_ln) if val > 0 else 0
                upper = math.exp(math.log(val) + 1.96 * se_ln) if val > 0 else 0

                obs = np.array([[ma, mb], [mc, md]])
                chi2_uncorrected, p_uncor, _, _ = chi2_contingency(obs, correction=False)
                chi2_yates, p_yates, _, _ = chi2_contingency(obs, correction=True)
                mid_p_val = calculate_mid_p(ma, mb, mc, md)

                st.markdown("---")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric(res_label, f"{val:.2f}")
                r2.metric("95% CI Lower", f"{lower:.3f}")
                r3.metric("95% CI Upper", f"{upper:.3f}")
                r4.metric("Mid-P exact", f"{max(mid_p_val, 0.0000001):.7f}")

                with st.container(border=True):
                    st.write(f"**Yates chi-square:** {chi2_yates:.3f}")
                    st.write(f"**Mid-P exact (2-tail):** {max(mid_p_val, 0.0000001):.7f}")
                    if mid_p_val < 0.05:
                        st.success("✨ มีนัยสำคัญทางสถิติ (p < 0.05)")
                    else:
                        st.warning("ยังไม่พบนัยสำคัญทางสถิติที่ระดับ p < 0.05")

                manual_res = f"Study Design: {manual_design}\n{res_label}: {val:.2f} (95% CI: {lower:.3f} - {upper:.3f})\nYates chi-square: {chi2_yates:.3f}\nMid-P exact: {max(mid_p_val, 0.0000001):.7f}"
                st.session_state['biv_man_res'] = manual_res
            except Exception as e:
                st.error(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ: {e}")
        else:
            st.warning("กรุณากรอกตัวเลขจำนวนในตาราง 2x2")

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
try:
    st.sidebar.image("odpc8_logo.png", use_container_width=True)
except Exception:
    st.sidebar.markdown("""
    <div class="app-brand">
        <div class="app-brand-title">🧬 Epi-Analytic Pro</div>
        <div class="app-brand-subtitle">ODPC8 Udon Thani • Privacy by Design</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "เลือกหัวข้อการวิเคราะห์",
    ["🏠 Dashboard",
     "👥 ประชากรและอัตราป่วย (Attack Rate)",
     "👤 บุคคล (Person)",
     "📊 สร้าง Epi Curve (Time)",
     "🗺️ สถานที่ (Place)",
     "🔬 Bivariate Analysis (OR/RR)",
     "🧬 Multiple Logistic Regression (AOR)",
     "🧪 Validation & Gold Standard"],
    key="main_menu_radio"
)

# ==========================================
# 5. DATA SOURCE & TEMPLATES
# ==========================================
df = None
st.sidebar.divider()
st.sidebar.subheader("💾 แหล่งข้อมูล (Data Source)")
st.sidebar.warning("อนุญาตเฉพาะข้อมูลจำลองหรือข้อมูลที่ตัดตัวระบุบุคคลแล้ว ห้ามอัปโหลดชื่อ เลขบัตรประชาชน HN, AN เบอร์โทรศัพท์ ที่อยู่ หรือรหัสที่ย้อนกลับไปหาผู้ป่วยได้")
confirmed = st.sidebar.checkbox(
    "ข้าพเจ้ายืนยันว่าข้อมูลถูก de-identify แล้ว",
    key="deidentified_confirmed"
)
uploaded_file = st.sidebar.file_uploader(
    "📂 เลือกไฟล์ Excel/CSV (ประมวลผลใน session)",
    type=['xlsx', 'csv'],
    disabled=not confirmed
)
if uploaded_file:
    candidate_df = load_data(uploaded_file)
    if candidate_df is not None:
        safe, risky_columns = validate_deidentified(candidate_df)
        if not safe:
            st.sidebar.error("ปฏิเสธไฟล์: พบคอลัมน์/ค่าที่อาจระบุตัวบุคคล")
            st.sidebar.code("\n".join(risky_columns))
        else:
            df = candidate_df

st.sidebar.markdown("---")
c_reset, c_clear = st.sidebar.columns(2)
if c_reset.button("🔄 เริ่มใหม่", use_container_width=True):
    st.cache_data.clear(); st.rerun()
if c_clear.button("🧹 ล้าง session", use_container_width=True):
    clear_analysis_state(); st.rerun()
if True:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📖 คู่มือการใช้งาน (Manual)")
    st.sidebar.markdown(f"""
    <div class="template-box" style="background-color: #FFF0F5; border-color: #E91E63;">
        <a class="template-link" href="https://docs.google.com/document/d/1AJe_OcKL1XSOsOdG2FTquoCWHi57iIQDSaEqEBWDYSA/edit?tab=t.0" target="_blank" style="font-size: 1.15rem; color: #D81B60 !important; font-weight: 600; text-align: center; margin-bottom: 0;">
            🖥️ เปิดคู่มือการใช้งานระบบ
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.subheader("📥 ไฟล์ตัวอย่าง (Templates)")
    st.sidebar.markdown(f"""
    <div class="template-box">
        <p style="margin-bottom:8px; font-size:1rem; color:#666;">ดาวน์โหลดไฟล์สำหรับทดลองระบบ:</p>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1WIK5KTiZMhUd_rpHg_YVzZl9gE5eDeOj/edit?usp=sharing&ouid=105507702062786209701&rtpof=true&sd=true" target="_blank">📄 1. Person/Time/Place</a>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1kZSskpErufY_9qTl-_1TZaVymGMnNikm/edit?usp=drive_link" target="_blank">🕒 2. Hourly Epidemic Curve</a>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1TPJDOoIWCiZBtsnXDlhcHcN5IM27TBOK/edit?usp=drive_link" target="_blank">🔬 3. Case Control Analysis</a>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1HR57-mVqo9TceAgF1tpzWvLQi662akzw/edit?usp=drive_link" target="_blank">📊 4. Cohort Study Analysis</a>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. MAIN CONTENT
# ==========================================

if True:
    if df is not None:
        total_n = len(df)
    else:
        total_n = 0
    render_hero(APP_NAME, "เครื่องมือวิเคราะห์เชิงพรรณนา เวลา สถานที่ และปัจจัยเสี่ยง สำหรับทีม SAT, SRRT และ JIT")
    st.caption(f"{APP_VERSION} • ไม่มีการเชื่อมต่อ Generative AI • ประมวลผลข้อมูลใน session")
    if df is not None and menu != "🏠 Dashboard":
        render_data_overview(df)
    elif df is None:
        st.caption("ยังไม่ได้เชื่อมต่อไฟล์ข้อมูล: เมนูที่ไม่ต้องใช้ไฟล์ เช่น Manual 2x2 สามารถใช้งานได้ทันที")

    # ------------------------------------------
    # 6.0 Dashboard
    # ------------------------------------------
    if menu == "🏠 Dashboard":
        if df is None:
            section_header("🏠", "Dashboard สรุปข้อมูลหลังนำเข้า", "อัปโหลด Excel/CSV ที่ผ่านการตัดตัวระบุบุคคลแล้ว")
            show_data_required_panel("Dashboard ต้องใช้ไฟล์ข้อมูล")
            st.info("หมายเหตุ: เมนู Bivariate Analysis > Manual 2x2 ยังใช้งานได้ทันที แม้ยังไม่อัปโหลดไฟล์")
            st.stop()
        render_dataset_dashboard(df)

    # ------------------------------------------
    # 6.1 Attack Rate
    # ------------------------------------------
    elif menu == "👥 ประชากรและอัตราป่วย (Attack Rate)":
        if df is None:
            section_header("👥", "ประชากรและอัตราป่วย (Attack Rate)", "คำนวณอัตราป่วยรวม และอัตราป่วยจำเพาะตามเพศ/กลุ่มอายุ")
            show_data_required_panel()
            st.stop()
        section_header("👥", "ประชากรและอัตราป่วย (Attack Rate)", "คำนวณอัตราป่วยรวม และอัตราป่วยจำเพาะตามเพศ/กลุ่มอายุ")
        sex_c = find_col(df, ['sex', 'gender', 'เพศ'])
        age_c = find_col(df, ['age', 'อายุ'])
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**ประชากรแยกตามเพศ**")
            pop_male = st.number_input("ประชากรชายทั้งหมด", min_value=1, value=100)
            pop_female = st.number_input("ประชากรหญิงทั้งหมด", min_value=1, value=100)
        with col_p2:
            st.markdown("**ประชากรแยกตามกลุ่มอายุ**")
            age_labels = ['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+']
            pop_age = {lbl: st.number_input(f"กลุ่ม {lbl}", min_value=0, value=0) for lbl in age_labels}

        if st.button("📈 คำนวณ"):
            total_pop = pop_male + pop_female
            ar = (total_n / total_pop * 100) if total_pop > 0 else 0
            st.metric("Overall Attack Rate", f"{ar:.2f} %")
            
            c_res1, c_res2 = st.columns(2)
            ar_sex_str, ar_age_str = "", ""
            with c_res1:
                st.markdown("**Sex-Specific Attack Rate**")
                if sex_c:
                    df['sex_temp'] = df[sex_c].astype(str).str.strip().replace({'1':'ชาย','2':'หญิง','1.0':'ชาย','2.0':'หญิง'})
                    m_case = len(df[df['sex_temp'] == 'ชาย'])
                    f_case = len(df[df['sex_temp'] == 'หญิง'])
                    ar_sex = pd.DataFrame({
                        "เพศ": ["ชาย", "หญิง"], "ป่วย (n)": [m_case, f_case], 
                        "ประชากร": [pop_male, pop_female], "AR (%)": [m_case/pop_male*100, f_case/pop_female*100]
                    })
                    st.table(ar_sex.style.format({"AR (%)": "{:.2f}"}))
                    ar_sex_str = ar_sex.to_string()
            with c_res2:
                st.markdown("**Age-Specific Attack Rate**")
                if age_c:
                    df['age_tmp'] = pd.cut(pd.to_numeric(df[age_c], errors='coerce'), bins=[0,5,15,25,35,45,55,65,120], labels=age_labels, right=False)
                    a_cases = df['age_tmp'].value_counts().reindex(age_labels, fill_value=0)
                    ar_age = [{"อายุ": l, "ป่วย": a_cases[l], "ประชากร": pop_age[l], "AR (%)": (a_cases[l]/pop_age[l]*100) if pop_age[l]>0 else 0} for l in age_labels]
                    ar_age_df = pd.DataFrame(ar_age)
                    st.table(ar_age_df.style.format({"AR (%)": "{:.2f}"}))
                    ar_age_str = ar_age_df.to_string()
            
            st.session_state['ar_context'] = f"Overall AR: {ar:.2f}%\nAR by Sex:\n{ar_sex_str}\nAR by Age:\n{ar_age_str}"


    # ------------------------------------------
    # 6.2 Descriptive Analysis
    # ------------------------------------------
    elif menu == "👤 บุคคล (Person)":
        if df is None:
            section_header("👤", "บุคคล (Person)", "สรุปเพศ กลุ่มอายุ อาการ และปัจจัยอื่น ๆ")
            show_data_required_panel()
            st.stop()
        section_header("👤", "บุคคล (Person)", "แสดงจำนวน ร้อยละ ตาราง และกราฟจำแนกตามลักษณะบุคคล")
        st.info(f"📋 จำนวนผู้ป่วยทั้งหมด (n) = {total_n} ราย")

        person_tables = {}
        c1, c2 = st.columns(2)
        with c1:
            sex_col = st.selectbox("ตัวแปรเพศ", df.columns)
            res_sex = frequency_table(df[sex_col], sex_col, total_n)
            person_tables["เพศ"] = res_sex
            st.dataframe(res_sex.style.format({'%': '{:.2f}'}), use_container_width=True, hide_index=True)
            fig_sex = px.pie(res_sex, names="กลุ่ม", values="n", hole=0.4,
                             title=f"สัดส่วนผู้ป่วยจำแนกตาม {sex_col}")
            fig_sex.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_sex, use_container_width=True, config=high_res_config)
        with c2:
            age_col = st.selectbox("ตัวแปรอายุ", df.columns)
            age_labels = ['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+']
            age_group = pd.cut(pd.to_numeric(df[age_col], errors='coerce'),
                               bins=[0,5,15,25,35,45,55,65,np.inf], labels=age_labels, right=False)
            age_counts = age_group.value_counts().reindex(age_labels, fill_value=0)
            res_age = pd.DataFrame({"ตัวแปร": age_col, "กลุ่ม": age_labels, "n": age_counts.values})
            res_age["%"] = np.where(total_n > 0, res_age["n"] / total_n * 100, 0)
            person_tables["กลุ่มอายุ"] = res_age
            st.dataframe(res_age.style.format({'%': '{:.2f}'}), use_container_width=True, hide_index=True)
            fig_age = px.bar(res_age, x="กลุ่ม", y="n", text="n",
                             title=f"จำนวนผู้ป่วยจำแนกตามกลุ่มอายุ ({age_col})",
                             color_discrete_sequence=['#6556FF'])
            fig_age.update_layout(xaxis_title="กลุ่มอายุ", yaxis_title="จำนวนผู้ป่วย")
            st.plotly_chart(fig_age, use_container_width=True, config=high_res_config)

        # --- ฟีเจอร์ใหม่: คำนวณค่าสถิติข้อมูลเชิงปริมาณ (Mean, Median, SD, Min, Max) ---
        st.markdown("---")
        st.subheader("📊 ค่าสถิติข้อมูลเชิงปริมาณ (Continuous Data)")
        
        default_idx = list(df.columns).index(age_col) if age_col in df.columns else 0
        num_col = st.selectbox("เลือกตัวแปรเพื่อคำนวณค่าสถิติ (เช่น อายุ, ระยะฟักตัว):", df.columns, index=default_idx)
        
        numeric_series = pd.to_numeric(df[num_col], errors='coerce').dropna()
        
        if not numeric_series.empty:
            mean_val = numeric_series.mean()
            median_val = numeric_series.median()
            sd_val = numeric_series.std()
            min_val = numeric_series.min()
            max_val = numeric_series.max()
            
            c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
            c_stat1.metric("ค่าเฉลี่ย (Mean)", f"{mean_val:.2f}")
            c_stat2.metric("มัธยฐาน (Median)", f"{median_val:.2f}")
            c_stat3.metric("ส่วนเบี่ยงเบนมาตรฐาน (SD)", f"{sd_val:.2f}")
            c_stat4.metric("พิสัย (Min - Max)", f"{min_val:.2f} - {max_val:.2f}")
            
            numeric_table = pd.DataFrame([{"ตัวแปร": num_col, "N valid": len(numeric_series),
                                           "Mean": mean_val, "Median": median_val, "SD": sd_val,
                                           "Min": min_val, "Max": max_val}])
            person_tables["สถิติเชิงปริมาณ"] = numeric_table
        else:
            st.warning("⚠️ ข้อมูลที่เลือกไม่สามารถคำนวณค่าทางสถิติได้ (กรุณาเลือกคอลัมน์ที่เป็นตัวเลข)")

        st.markdown("---")
        st.subheader("อาการแสดง (1=มีอาการ)")
        symp_cols = st.multiselect("เลือกตัวแปรอาการ", df.columns)
        if symp_cols:
            s_df = pd.DataFrame([{"ตัวแปร": "อาการ", "กลุ่ม": c, "n": (pd.to_numeric(df[c], errors='coerce')==1).sum(),
                                  "%": (pd.to_numeric(df[c], errors='coerce')==1).sum()/total_n*100}
                                 for c in symp_cols]).sort_values("%", ascending=True)
            person_tables["อาการ"] = s_df
            
            fig_s = px.bar(s_df, x="%", y="กลุ่ม", orientation='h', text_auto='.1f', color_discrete_sequence=['#E91E63'])
            fig_s.update_layout(font=dict(family="Sarabun", size=16, color="#4A4A4A"), title="แผนภูมิแท่งแนวนอนแสดงร้อยละของอาการ")
            
            st.plotly_chart(fig_s, use_container_width=True, config=high_res_config)
            st.caption("📸 คลิกที่ไอคอนกล้องถ่ายรูปมุมขวาบนของแผนภูมิแท่ง เพื่อดาวน์โหลดรูปภาพความละเอียดสูง")

        st.markdown("---")
        st.subheader("🧩 ตัวแปรปัจจัยอื่น ๆ")
        factor_cols = st.multiselect(
            "เลือกตัวแปร เช่น อาหาร การวินิจฉัย ความรุนแรง อาชีพ หรือปัจจัยสัมผัส",
            [c for c in df.columns if c not in {sex_col, age_col}],
            key="person_other_factors"
        )
        for i, factor_col in enumerate(factor_cols):
            factor_table = frequency_table(df[factor_col], factor_col, total_n)
            person_tables[f"ปัจจัย_{i+1}"] = factor_table
            st.markdown(f"#### {factor_col}")
            fc1, fc2 = st.columns([1, 1.25])
            with fc1:
                st.dataframe(factor_table.style.format({'%': '{:.2f}'}), use_container_width=True, hide_index=True)
            with fc2:
                factor_fig = px.bar(factor_table, x="กลุ่ม", y="n", text="n",
                                    title=f"จำนวนผู้ป่วยจำแนกตาม {factor_col}",
                                    color_discrete_sequence=['#00B4D8'])
                factor_fig.update_layout(xaxis_title=factor_col, yaxis_title="จำนวนผู้ป่วย")
                st.plotly_chart(factor_fig, use_container_width=True, config=high_res_config)

        if person_tables:
            st.markdown("---")
            st.subheader("📥 ส่งออกผลการวิเคราะห์แบบสรุป")
            combined_person = pd.concat(person_tables.values(), ignore_index=True, sort=False)
            ec1, ec2 = st.columns(2)
            ec1.download_button("⬇️ ดาวน์โหลด Person Analysis (Excel)",
                                tables_to_excel_bytes(person_tables),
                                "person_analysis.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)
            ec2.download_button("⬇️ ดาวน์โหลด Person Analysis (CSV)",
                                combined_person.to_csv(index=False).encode("utf-8-sig"),
                                "person_analysis.csv", "text/csv", use_container_width=True)


    # ------------------------------------------
    # 6.3 Epidemic Curve 
    # ------------------------------------------
    elif menu == "📊 สร้าง Epi Curve (Time)":
        if df is None:
            section_header("📊", "Interactive Epidemic Curve", "สร้างเส้นโค้งการระบาดแบบโต้ตอบ พร้อมกำหนดช่วงเวลาและกลุ่มสี")
            show_data_required_panel()
            st.stop()
        section_header("📊", "Interactive Epidemic Curve", "สร้างเส้นโค้งการระบาดแบบโต้ตอบ พร้อมกำหนดช่วงเวลาและกลุ่มสี")

        st.markdown(
            """
            <div class="template-box" style="background: linear-gradient(135deg, #FFF0F5 0%, #ffffff 100%); border-left: 5px solid #E91E63; padding: 15px; margin-bottom: 25px;">
                <span style="font-weight: 600; color: #880E4F;">💡 แนวทางการแบ่งช่วงเวลา (Bin):</span> 
                ท่านสามารถคลิกดูแนวทางการตั้งค่าและแบ่ง Bin ให้สอดคล้องกับชนิดของโรค/เชื้อโรค โดยใช้ปัญญาประดิษฐ์วิเคราะห์ได้ที่ 
                <a href="https://script.google.com/macros/s/AKfycbycVxWwodTKeNV-xayinlxkx1SmYkhjeFsjPHBSinshVTKtbsZCMuYpTKI9oFmqUnn_/exec" target="_blank" style="color: #D81B60; font-weight: 600; text-decoration: underline;">
                    ระบบรายงานและวิเคราะห์ข้อมูลระบาดวิทยา (Epidemiology Dashboard)
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Auto-select the likely onset-date column so the running-number column (ลำดับ)
        # will not be selected accidentally.
        likely_date_keywords = ["วันเริ่มป่วย", "onset", "date_onset", "date onset", "เริ่มป่วย"]
        default_date_idx = 0
        for i, col in enumerate(df.columns):
            col_text = str(col).strip().lower()
            if any(k.lower() in col_text for k in likely_date_keywords):
                default_date_idx = i
                break

        date_col = st.sidebar.selectbox("คอลัมน์วันเริ่มป่วย", df.columns, index=default_date_idx)
        col_grp = st.sidebar.selectbox("ตัวแปรแยกกลุ่มสี:", ["<none>"] + list(df.columns))

        custom_color = st.sidebar.color_picker("🎨 เลือกสีแผนภูมิแท่งหลัก", "#E91E63")

        unit_map = {"Hour": "h", "Day": "d", "Week": "W", "Month": "ME", "30 Min": "30min"}
        bin_unit = st.sidebar.selectbox("หน่วยเวลา", list(unit_map.keys()), index=0)
        bin_size = st.sidebar.number_input("ขนาด Bin", min_value=1, value=1)
        freq = f"{bin_size}{unit_map[bin_unit]}"

        pad_before = st.sidebar.number_input(f"เพิ่มช่วงว่างก่อนหน้า ({bin_unit})", value=1)
        pad_after = st.sidebar.number_input(f"เพิ่มช่วงว่างข้างหลัง ({bin_unit})", value=1)

        # Never overwrite the original selected column. Use a dedicated parsed column.
        df_plot = df.copy()
        original_date_col = f"{date_col} (ข้อมูลเดิม)"
        df_plot[original_date_col] = df_plot[date_col]
        df_plot["_onset_datetime_ce"] = parse_epi_date_series(df_plot[date_col])
        df_clean = df_plot.dropna(subset=["_onset_datetime_ce"]).copy()

        with st.expander("🔎 ตรวจสอบรูปแบบวันที่ที่ระบบอ่านได้", expanded=True):
            preview_dates = df_plot[[original_date_col, "_onset_datetime_ce"]].copy()
            preview_dates.columns = ["วันที่ในไฟล์เดิม", "วันที่หลังแปลงเป็น ค.ศ."]
            preview_dates["วันที่หลังแปลงเป็น ค.ศ."] = preview_dates["วันที่หลังแปลงเป็น ค.ศ."].dt.strftime("%d/%m/%Y %H:%M").fillna("อ่านวันที่ไม่ได้")
            st.dataframe(preview_dates.head(20), use_container_width=True)
            invalid_n = int(df_plot["_onset_datetime_ce"].isna().sum())
            valid_n = int(df_plot["_onset_datetime_ce"].notna().sum())
            st.caption(f"อ่านวันที่ได้ {valid_n} รายการ | อ่านวันที่ไม่ได้ {invalid_n} รายการ | คอลัมน์ที่เลือก: {date_col}")

        if not df_clean.empty:
            min_dt, max_dt = df_clean["_onset_datetime_ce"].min(), df_clean["_onset_datetime_ce"].max()

            if "h" in freq or "min" in freq:
                start_range = (min_dt - pd.Timedelta(hours=pad_before)).floor('h')
                end_range = (max_dt + pd.Timedelta(hours=pad_after)).ceil('h')
            else:
                start_range = (min_dt - pd.to_timedelta(pad_before, unit='d')).floor('d')
                end_range = (max_dt + pd.to_timedelta(pad_after, unit='d')).ceil('d')

            full_range = pd.date_range(start=start_range, end=end_range, freq=freq)

            if col_grp == "<none>":
                counts = df_clean.groupby(pd.Grouper(key="_onset_datetime_ce", freq=freq)).size()
                chart_df = counts.reindex(full_range, fill_value=0).reset_index()
                chart_df.columns = ["Onset Date/Time", "Cases"]
                fig = px.bar(chart_df, x="Onset Date/Time", y="Cases", text_auto=True, color_discrete_sequence=[custom_color])
            else:
                counts = df_clean.groupby([pd.Grouper(key="_onset_datetime_ce", freq=freq), col_grp]).size().unstack(fill_value=0)
                chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name="Cases")
                chart_df.columns = ["Onset Date/Time", col_grp, "Cases"]
                fig = px.bar(chart_df, x="Onset Date/Time", y="Cases", color=col_grp, color_discrete_sequence=px.colors.sequential.RdPu[::-1])

            fig.update_layout(
                font=dict(family="Sarabun", size=16, color="#4A4A4A"),
                title="แผนภูมิแท่งแสดงการกระจายตัวของผู้ป่วยตามเวลาเริ่มป่วย (Epidemic Curve)",
                bargap=0.01, 
                xaxis=dict(type='date', tickformat='%d/%m/%Y %H:%M'),
                xaxis_title="Onset Date/Time",
                yaxis_title="Number of Cases",
                hovermode="x unified"
            )
            fig.update_traces(marker_line_width=0.5, marker_line_color='white')

            st.plotly_chart(fig, use_container_width=True, config=high_res_config)
            st.caption("📸 คลิกที่ไอคอนกล้องถ่ายรูปมุมขวาบนของแผนภูมิแท่ง เพื่อดาวน์โหลดรูปภาพความละเอียดสูง")

        else:
            st.error("❌ ไม่สามารถวิเคราะห์ได้ เนื่องจากรูปแบบวันที่ในไฟล์ไม่ถูกต้อง หรือเลือกคอลัมน์วันที่ไม่ถูกต้อง")
            st.info("โปรดตรวจสอบว่าเลือกคอลัมน์วันเริ่มป่วย เช่น 'วันเริ่มป่วย' ไม่ใช่คอลัมน์ลำดับ/รหัสผู้ป่วย")

    # ------------------------------------------
    # 6.4 Spot Map
    # ------------------------------------------

    elif menu == "🗺️ สถานที่ (Place)":
        if df is None:
            section_header("🗺️", "สถานที่ (Place)", "แสดง Spot Map และตารางจำแนกตามสถานที่")
            show_data_required_panel()
            st.stop()
        section_header("🗺️", "สถานที่ (Place)", "แสดงตำแหน่งผู้ป่วยและการกระจายตามหมู่ ชั้นเรียน ห้อง หรือสถานที่อื่น")
        lat_c = next((c for c in df.columns if any(p in c.lower() for p in ['lat', 'latitude', 'ละติจูด'])), None)
        lon_c = next((c for c in df.columns if any(p in c.lower() for p in ['lon', 'longitude', 'ลองจิจูด'])), None)
        
        if lat_c and lon_c:
            df_m = df.copy()
            df_m[lat_c] = pd.to_numeric(df_m[lat_c], errors="coerce")
            df_m[lon_c] = pd.to_numeric(df_m[lon_c], errors="coerce")
            df_m = df_m.dropna(subset=[lat_c, lon_c]).copy()

        if lat_c and lon_c and not df_m.empty:

            st.sidebar.markdown("---")
            st.sidebar.subheader("⚙️ ตั้งค่าแผนที่")
            
            precision_mode = st.sidebar.radio(
                "ความละเอียดตำแหน่ง",
                ["ปกปิดตำแหน่งด้วย jitter (แนะนำ)", "ตำแหน่งจริงเฉพาะ session นี้"],
                help="ระบบไม่บันทึกพิกัดลงฐานข้อมูล แต่แผนที่ฐานอาจมีการร้องขอ tile ผ่านอินเทอร์เน็ต"
            )
            jitter_meters = st.sidebar.slider("ระยะ jitter สูงสุด (เมตร)", 10, 200, 30, 10,
                                               disabled="ตำแหน่งจริง" in precision_mode)
            
            buffer_radius = st.sidebar.number_input("รัศมีควบคุมโรค (เมตร)", min_value=0, value=100, step=50)
            map_type = st.sidebar.radio("รูปแบบแผนที่", ["ดาวเทียม (Google Hybrid)", "แผนที่ถนน (OpenStreetMap)"])

            if map_type == "ดาวเทียม (Google Hybrid)":
                tiles_url = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
                attr = 'Google'
            else:
                tiles_url = 'OpenStreetMap'
                attr = 'OpenStreetMap'

            if "jitter" in precision_mode:
                map_df = jitter_coordinates(df_m, lat_c, lon_c, jitter_meters,
                                            seed=int(st.session_state.session_nonce[:8], 16))
                plot_lat, plot_lon = '_masked_lat', '_masked_lon'
            else:
                map_df = df_m.copy()
                plot_lat, plot_lon = lat_c, lon_c
                st.warning("กำลังแสดงตำแหน่งจริงบนหน้าจอเฉพาะ session นี้ ระบบไม่จัดเก็บหรือส่งออกพิกัด แต่ควรใช้เฉพาะในสภาพแวดล้อมที่ควบคุมได้")

            m = folium.Map(
                location=[map_df[plot_lat].mean(), map_df[plot_lon].mean()], 
                zoom_start=16, 
                tiles=tiles_url, 
                attr=attr
            )

            for idx, r in map_df.iterrows():
                if buffer_radius > 0:
                    folium.Circle(
                        location=[r[plot_lat], r[plot_lon]], 
                        radius=buffer_radius, 
                        color='#FFEB3B', 
                        weight=2,
                        fill=True,
                        fill_opacity=0.25,
                        fill_color='#FF9800'
                    ).add_to(m)

                marker = folium.CircleMarker(
                    location=[r[plot_lat], r[plot_lon]], 
                    radius=6, 
                    color='#E91E63',
                    fill=True, 
                    fill_opacity=1.0
                )
                
                # Deliberately no popup/tooltip: prevent row-level disclosure.
                marker.add_to(m)

            components.html(m._repr_html_(), height=650)
            st.caption("💡 แนะนำให้ใช้ฟังก์ชัน Screen Capture (Print Screen) ของคอมพิวเตอร์ เพื่อบันทึกภาพแผนที่")

        else: 
            st.info("ℹ️ ไม่พบพิกัด Lat/Lon หรือพิกัดไม่มีข้อมูล จึงข้าม Spot Map แต่ยังวิเคราะห์ตัวแปรสถานที่ด้านล่างได้")

        st.markdown("---")
        st.subheader("📍 ตารางวิเคราะห์ผู้ป่วยจำแนกตามสถานที่")
        location_candidates = [c for c in df.columns if c not in {lat_c, lon_c} and c not in detect_pii_columns(df)]
        location_cols = st.multiselect(
            "เลือกตัวแปรสถานที่ เช่น หมู่บ้าน หมู่ ชั้นเรียน ห้อง โรงเรียน จุดรับประทานอาหาร หรือพื้นที่เกิดเหตุ",
            location_candidates,
            key="place_variables"
        )
        place_tables = {}
        for i, location_col in enumerate(location_cols):
            place_table = frequency_table(df[location_col], location_col, total_n)
            place_tables[f"สถานที่_{i+1}"] = place_table
            st.markdown(f"#### {location_col}")
            pc1, pc2 = st.columns([1, 1.25])
            with pc1:
                st.dataframe(place_table.style.format({'%': '{:.2f}'}), use_container_width=True, hide_index=True)
            with pc2:
                place_fig = px.bar(place_table, x="กลุ่ม", y="n", text="n",
                                   title=f"จำนวนผู้ป่วยจำแนกตาม {location_col}",
                                   color_discrete_sequence=['#10B981'])
                place_fig.update_layout(xaxis_title=location_col, yaxis_title="จำนวนผู้ป่วย")
                st.plotly_chart(place_fig, use_container_width=True, config=high_res_config)

        if place_tables:
            combined_place = pd.concat(place_tables.values(), ignore_index=True, sort=False)
            pcsv, pxlsx = st.columns(2)
            pxlsx.download_button("⬇️ ดาวน์โหลด Place Analysis (Excel)",
                                  tables_to_excel_bytes(place_tables),
                                  "place_analysis.xlsx",
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  use_container_width=True)
            pcsv.download_button("⬇️ ดาวน์โหลด Place Analysis (CSV)",
                                 combined_place.to_csv(index=False).encode("utf-8-sig"),
                                 "place_analysis.csv", "text/csv", use_container_width=True)

    # ------------------------------------------
    # 6.5 Bivariate Analysis
    # ------------------------------------------
    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        section_header("🔬", "Bivariate Analysis & 2x2 Table", "คำนวณ OR/RR, 95% CI และ Mid-P exact สำหรับปัจจัยเสี่ยง")

        tab1, tab2 = st.tabs(["📁 วิเคราะห์จากไฟล์ข้อมูล", "🔢 กรอกข้อมูลเอง (Manual 2x2)"])

        with tab1:
            st.subheader("📁 วิเคราะห์ปัจจัยเสี่ยงจากไฟล์ที่อัปโหลด")
            if df is None:
                st.warning("เมนูวิเคราะห์จากไฟล์ต้องอัปโหลด Excel/CSV ที่ผ่าน PII screening ก่อน แต่ Manual 2x2 ใช้ได้ทันที")
            else:
                out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns, key="file_out")
                outcome_levels = list(pd.Series(df[out_v].dropna().unique()).astype(str))
                out_positive = st.selectbox("ค่าที่หมายถึงป่วย/เกิดเหตุการณ์ (Outcome=1)", outcome_levels)
                out_negative = st.selectbox("ค่าที่หมายถึงไม่ป่วย/ไม่เกิดเหตุการณ์ (Outcome=0)",
                                            [x for x in outcome_levels if x != out_positive])
                design = st.radio("ประเภทการศึกษา", ["Case-control Study (OR)", "Cohort Study (RR)"], key="file_design")
                exp_list = st.multiselect("เลือกปัจจัยเสี่ยง", [c for c in df.columns if c != out_v], key="file_exp")
                st.caption("สำหรับ exposure ให้กำหนดค่า 1=สัมผัส และ 0=ไม่สัมผัสในไฟล์ ระบบจะไม่เดา coding ให้อัตโนมัติ")

                if st.button("🚀 ประมวลผลจากไฟล์"):
                    results = []
                    for exp_v in exp_list:
                        n_before = len(df)
                        temp = df[[out_v, exp_v]].copy().dropna()
                        temp[out_v] = temp[out_v].astype(str).map({str(out_positive): 1, str(out_negative): 0})
                        temp[exp_v] = pd.to_numeric(temp[exp_v], errors='coerce')
                        temp = temp[temp[out_v].isin([1, 0]) & temp[exp_v].isin([1, 0])]
                        n_after = len(temp)

                        if len(temp) > 0:
                            a = len(temp[(temp[exp_v]==1) & (temp[out_v]==1)])
                            b = len(temp[(temp[exp_v]==1) & (temp[out_v]==0)])
                            c = len(temp[(temp[exp_v]==0) & (temp[out_v]==1)])
                            d = len(temp[(temp[exp_v]==0) & (temp[out_v]==0)])

                            try:
                                zero_cell = any(v == 0 for v in [a, b, c, d])
                                aa, bb, cc, dd = (a+.5, b+.5, c+.5, d+.5) if zero_cell else (a, b, c, d)
                                if "Case-control" in design:
                                    m_label = "OR"
                                    measure = (aa * dd) / (bb * cc)
                                    se_ln = math.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
                                else:
                                    m_label = "RR"
                                    measure = (aa / (aa + bb)) / (cc / (cc + dd))
                                    se_ln = math.sqrt((1/aa - 1/(aa+bb)) + (1/cc - 1/(cc+dd)))

                                ci_l = math.exp(math.log(measure) - 1.96 * se_ln) if measure > 0 else 0
                                ci_u = math.exp(math.log(measure) + 1.96 * se_ln) if measure > 0 else 0

                                mid_p_val = calculate_mid_p(a, b, c, d)

                                results.append({
                                    "ปัจจัย": exp_v, 
                                    "N ก่อน/หลังตัด missing": f"{n_before}/{n_after}",
                                    "ป่วย(+)": a, "ไม่ป่วย(+)": b, 
                                    "ป่วย(-)": c, "ไม่ป่วย(-)": d, 
                                    m_label: measure, 
                                    "95% CI Lower": ci_l, 
                                    "95% CI Upper": ci_u, 
                                    "Mid-P (2-tail)": max(mid_p_val, 0),
                                    "Zero-cell correction": "Haldane-Anscombe +0.5" if zero_cell else "ไม่ใช้"
                                })
                            except: pass

                    if results:
                        res_df = pd.DataFrame(results)
                        st.success(f"✅ ประมวลผลสำเร็จ (ใช้สูตร Taylor Series และ Mid-P ตามมาตรฐาน OpenEpi)")
                        st.dataframe(res_df.style.format({
                            m_label: "{:.2f}", 
                            "95% CI Lower": "{:.3f}", 
                            "95% CI Upper": "{:.3f}", 
                            "Mid-P (2-tail)": "{:.7f}"
                        }))
                        st.session_state['biv_file_res'] = res_df.to_string()
                        st.session_state.research_results['bivariate'] = res_df
                        export_df = safe_export(res_df)
                        st.download_button("⬇️ ดาวน์โหลดผลลัพธ์ที่ไม่ระบุตัวบุคคล (CSV)",
                                           export_df.to_csv(index=False).encode('utf-8-sig'),
                                           "bivariate_results.csv", "text/csv")
                        st.warning("ผลวิเคราะห์แสดงความสัมพันธ์ ไม่ได้ยืนยันความเป็นเหตุ–ผล")
                    else:
                        st.warning("⚠️ ไม่พบข้อมูลที่เพียงพอในการวิเคราะห์")

        with tab2:
            render_manual_2x2_calculator()

    # ------------------------------------------
    # 6.6 Logistic Regression
    # ------------------------------------------
    elif menu == "🧬 Multiple Logistic Regression (AOR)":
        if df is None:
            section_header("🧬", "Multiple Logistic Regression", "วิเคราะห์ Adjusted OR โดยควบคุมตัวแปรกวน")
            show_data_required_panel()
            st.stop()
        section_header("🧬", "Multiple Logistic Regression", "วิเคราะห์ Adjusted OR โดยควบคุมตัวแปรกวน")
        out_v = st.selectbox("Outcome", df.columns, key="mlr_out")
        mlr_levels = list(pd.Series(df[out_v].dropna().unique()).astype(str))
        mlr_positive = st.selectbox("ค่าที่หมายถึงเกิดเหตุการณ์ (Outcome=1)", mlr_levels, key="mlr_pos")
        mlr_negative = st.selectbox("ค่าที่หมายถึงไม่เกิดเหตุการณ์ (Outcome=0)",
                                    [x for x in mlr_levels if x != mlr_positive], key="mlr_neg")
        exp_v = st.selectbox("ปัจจัยหลัก", [c for c in df.columns if c != out_v])
        adj_v = st.multiselect("ตัวแปรกวน", [c for c in df.columns if c not in [out_v, exp_v]])
        st.caption("ตัวแปรอิสระต้องเป็นตัวเลขหรือ binary 0/1 และ Outcome จะเข้ารหัสตามค่าที่เลือกด้านบน")
        
        if st.button("🚀 คำนวณ AOR"):
            try:
                n_before = len(df)
                df_m = df[[out_v, exp_v] + adj_v].copy().dropna()
                df_m[out_v] = df_m[out_v].astype(str).map({str(mlr_positive): 1, str(mlr_negative): 0})
                for c in [exp_v] + adj_v:
                    df_m[c] = pd.to_numeric(df_m[c], errors='coerce')
                df_m = df_m.dropna()
                n_after = len(df_m)
                events = int(df_m[out_v].sum())
                non_events = int(len(df_m) - events)
                parameter_count = 1 + len(adj_v)
                if df_m[out_v].nunique() != 2:
                    raise ValueError("Outcome ต้องมีทั้งกลุ่ม 0 และ 1")
                if events < 10 * parameter_count or non_events < 10 * parameter_count:
                    st.warning(f"จำนวนเหตุการณ์/ไม่เกิดเหตุการณ์ ({events}/{non_events}) ต่ำกว่าเกณฑ์แนะนำ 10 ต่อพารามิเตอร์ ({parameter_count} พารามิเตอร์) ผลอาจไม่เสถียร")

                x_for_vif = sm.add_constant(df_m[[exp_v] + adj_v].astype(float), has_constant='add')
                vif_df = pd.DataFrame({
                    "ตัวแปร": x_for_vif.columns,
                    "VIF": [variance_inflation_factor(x_for_vif.values, i) for i in range(x_for_vif.shape[1])]
                })
                vif_df = vif_df[vif_df["ตัวแปร"] != "const"]
                if (vif_df["VIF"] >= 10).any():
                    st.warning("พบ multicollinearity สูง (VIF ≥ 10) กรุณาทบทวนตัวแปร")
                
                formula = f"Q('{out_v}') ~ Q('{exp_v}')"
                if adj_v: formula += " + " + " + ".join([f"Q('{a}')" for a in adj_v])
                
                model = smf.logit(formula, data=df_m).fit(disp=0)
                converged = bool(model.mle_retvals.get('converged', False))
                if not converged:
                    raise ValueError("แบบจำลองไม่ convergence; อาจเกิด complete/quasi-complete separation")
                predicted = model.predict(df_m)
                if ((predicted < 1e-6) | (predicted > 1-1e-6)).any():
                    st.warning("ตรวจพบค่าพยากรณ์ใกล้ 0 หรือ 1 มาก อาจมี complete/quasi-complete separation")
                
                conf_int = model.conf_int()
                res_df = pd.DataFrame({
                    "Factors": model.params.index,
                    "Adjusted OR (AOR)": np.exp(model.params.values),
                    "95% CI Lower": np.exp(conf_int[0].values),
                    "95% CI Upper": np.exp(conf_int[1].values),
                    "P-value": model.pvalues.values
                })

                res_df = res_df[res_df['Factors'] != 'Intercept']
                res_df['Factors'] = res_df['Factors'].str.extract(r"Q\('(.*)'\)")[0].fillna(res_df['Factors'])

                st.subheader("📋 สรุปผลการวิเคราะห์ปัจจัยเสี่ยง")
                st.info(f"N ก่อน/หลังตัด missing: {n_before}/{n_after} | Events/Non-events: {events}/{non_events} | Converged: {converged}")
                st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}), use_container_width=True)
                st.dataframe(res_df.style.format({
                    "Adjusted OR (AOR)": "{:.2f}",
                    "95% CI Lower": "{:.2f}",
                    "95% CI Upper": "{:.2f}",
                    "P-value": "{:.4f}"
                }).apply(lambda x: ['background-color: #F8BBD0' if x['P-value'] < 0.05 else '' for _ in x], axis=1), 
                use_container_width=True)
                
                st.success("✅ คำนวณค่า Adjusted OR และ 95% CI สำเร็จ")
                st.session_state['mlr_res'] = res_df.to_string()
                st.session_state.research_results['logistic'] = res_df
                st.code(formula, language="text")
                st.caption(f"Python {platform.python_version()} • pandas {pd.__version__} • statsmodels {statsmodels.__version__}")
                st.warning("ผลวิเคราะห์แสดงความสัมพันธ์ ไม่ได้ยืนยันความเป็นเหตุ–ผล")
                st.download_button("⬇️ ดาวน์โหลดผล AOR ที่ไม่ระบุตัวบุคคล (CSV)",
                                   safe_export(res_df).to_csv(index=False).encode('utf-8-sig'),
                                   "adjusted_or_results.csv", "text/csv")

            except Exception as e:
                st.error(f"⚠️ ไม่สามารถประมวลผลได้: {e}")

    elif menu == "🧪 Validation & Gold Standard":
        section_header("🧪", "Validation เทียบ Gold Standard", "Bland–Altman, ICC(A,1) และชุดข้อมูลจำลองมาตรฐาน")
        st.info("อัปโหลดตารางผลลัพธ์แบบไม่ระบุตัวบุคคล โดยแต่ละแถวเป็นค่าจากโจทย์ทดสอบเดียวกัน")
        validation_file = st.file_uploader("CSV/XLSX: ผลจากระบบและ Gold Standard", type=['csv', 'xlsx'], key="validation_file")
        if validation_file:
            val_df = load_data(validation_file)
            system_col = st.selectbox("คอลัมน์ผลจาก Epi-Analytic Pro", val_df.columns)
            reference_col = st.selectbox("คอลัมน์ผลจาก Gold Standard", [c for c in val_df.columns if c != system_col])
            pair = val_df[[system_col, reference_col]].apply(pd.to_numeric, errors='coerce').dropna()
            if len(pair) >= 3:
                diff = pair[system_col] - pair[reference_col]
                mean_pair = pair.mean(axis=1)
                bias = diff.mean()
                sd_diff = diff.std(ddof=1)
                lower, upper = bias - 1.96 * sd_diff, bias + 1.96 * sd_diff
                icc = icc_absolute_agreement(pair[[system_col, reference_col]].values)
                result = pd.DataFrame([{
                    "N pairs": len(pair), "Mean bias": bias,
                    "95% LoA lower": lower, "95% LoA upper": upper,
                    "ICC(A,1)": icc, "ผ่านเกณฑ์ ICC > 0.95": bool(icc > .95)
                }])
                st.dataframe(result.style.format({"Mean bias":"{:.6f}", "95% LoA lower":"{:.6f}",
                                                  "95% LoA upper":"{:.6f}", "ICC(A,1)":"{:.4f}"}),
                             use_container_width=True)
                ba_df = pd.DataFrame({"Mean of methods": mean_pair, "Difference": diff,
                                      "Bias": bias, "Lower LoA": lower, "Upper LoA": upper})
                fig_ba = px.scatter(ba_df, x="Mean of methods", y="Difference", title="Bland–Altman Plot")
                for y, label, color in [(bias, "Bias", "#6556FF"), (lower, "Lower LoA", "#EC4899"), (upper, "Upper LoA", "#EC4899")]:
                    fig_ba.add_hline(y=y, annotation_text=label, line_color=color, line_dash="dash")
                st.plotly_chart(fig_ba, use_container_width=True, config=high_res_config)
                export = pd.concat([result, ba_df], axis=0, ignore_index=True)
                st.download_button("⬇️ Export Bland–Altman และ ICC", export.to_csv(index=False).encode('utf-8-sig'),
                                   "validation_bland_altman_icc.csv", "text/csv")
            else:
                st.warning("ต้องมีคู่ข้อมูลตัวเลขอย่างน้อย 3 คู่")

        synthetic = make_synthetic_scenarios()
        st.download_button("⬇️ ชุดข้อมูลจำลอง 3 สถานการณ์", synthetic.to_csv(index=False).encode('utf-8-sig'),
                           "HE69-085_synthetic_scenarios.csv", "text/csv")
        st.caption("สถานการณ์: cohort สมดุล, case-control ที่มี confounding/missing และ small sample ที่มี zero cell")

# --- Footer ---
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #880E4F;'>{APP_NAME} | {APP_VERSION}<br>พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี กรมควบคุมโรค</div>", unsafe_allow_html=True)
