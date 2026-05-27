import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import timedelta
import scipy.stats as stats
from scipy.stats import hypergeom, chi2_contingency
import plotly.express as px
import folium
from streamlit_folium import folium_static
import requests
import math
import re
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & STYLING (MODERN SARABUN)
# ==========================================
st.set_page_config(
    page_title="Epi-Analytic Pro ODPC8", 
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
# 2. SESSION STATE & AI HELPER
# ==========================================
if 'registered' not in st.session_state:
    st.session_state['registered'] = False

def generate_ai_summary(api_key, context_text, menu_name):
    api_key = api_key or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ กรุณาระบุ Gemini API Key ในแถบเมนูด้านซ้ายเพื่อเปิดใช้งานผู้ช่วย AI"
    try:
        genai.configure(api_key=api_key)
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not valid_models: return "❌ API Key ของท่านไม่มีสิทธิ์ใช้งานโมเดลใดๆ"
        target_model = next((m for m in valid_models if '1.5-flash' in m), valid_models[0])
        
        model = genai.GenerativeModel(target_model)
        prompt = f"""
        คุณคือนักระบาดวิทยาผู้เชี่ยวชาญ กรุณาสรุปผลการวิเคราะห์ข้อมูลต่อไปนี้จากเมนู '{menu_name}' 
        เพื่อนำไปเขียนในรายงานการสอบสวนการระบาดของโรค (ขอแบบสั้น กระชับ เป็นทางการ ตรงประเด็น)
        
        ข้อมูลสถิติที่ประมวลผลได้:
        {context_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ ไม่สามารถเชื่อมต่อ AI ได้: {e}"

high_res_config = {
    'displaylogo': False,
    'toImageButtonOptions': {'format': 'png', 'filename': 'Epi_Chart_Export', 'height': 720, 'width': 1280, 'scale': 2}
}


def render_hero(title, subtitle, kicker="AI DATA ANALYTICS • ODPC8"):
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

def find_col(df, possible_names):
    return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
try:
    st.sidebar.image("odpc8_logo.png", use_container_width=True)
except Exception:
    st.sidebar.markdown("""
    <div class="app-brand">
        <div class="app-brand-title">🧬 Epi-Analytic Pro</div>
        <div class="app-brand-subtitle">ODPC8 Udon Thani • AI Data Analytics</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนเพื่อปลดล็อกเมนูวิเคราะห์")
else:
    st.sidebar.subheader("🤖 ผู้ช่วย AI สรุปผล")
    api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="แนะนำให้เก็บใน .streamlit/secrets.toml หรือ Streamlit Cloud Secrets")
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "เลือกหัวข้อการวิเคราะห์", 
        ["👥 ประชากรและอัตราป่วย (Attack Rate)",
         "👤 พรรณนา (Descriptive)", 
         "📊 สร้าง Epi Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Bivariate Analysis (OR/RR)", 
         "🧬 Multiple Logistic Regression (AOR)",
         "📝 ข้อมูลการลงทะเบียน (แก้ไข)"],
        key="main_menu_radio" 
    )

# ==========================================
# 5. DATA SOURCE & TEMPLATES
# ==========================================
df = None
if st.session_state['registered']:
    st.sidebar.divider()
    st.sidebar.subheader("💾 แหล่งข้อมูล (Data Source)")
    source_choice = st.sidebar.radio("เลือกแหล่งข้อมูล:", ["อัปโหลดไฟล์ (Excel/CSV)", "Google Sheets"], key="data_source_radio")
    
    if source_choice == "อัปโหลดไฟล์ (Excel/CSV)":
        uploaded_file = st.sidebar.file_uploader("📂 เลือกไฟล์ข้อมูล", type=['xlsx', 'csv'])
        if uploaded_file: df = load_data(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input("🔗 ลิงก์ Google Sheets:")
        if sheet_url:
            try:
                # Read public Google Sheets without the external streamlit-gsheets package.
                # The sheet must be shared as "Anyone with the link" or published/exportable as CSV.
                if "docs.google.com/spreadsheets" in sheet_url:
                    match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
                    if match:
                        sheet_id = match.group(1)
                        gid_match = re.search(r'gid=([0-9]+)', sheet_url)
                        gid = gid_match.group(1) if gid_match else "0"
                        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
                        df = pd.read_csv(csv_url)
                    else:
                        st.error("ไม่พบ Google Sheet ID จากลิงก์ที่ระบุ")
                elif sheet_url.lower().endswith('.csv') or 'format=csv' in sheet_url.lower():
                    df = pd.read_csv(sheet_url)
                else:
                    st.error("กรุณาใช้ลิงก์ Google Sheets หรือ CSV URL ที่เปิดสิทธิ์แชร์แล้ว")
                    
                if st.sidebar.button("🔄 อัปเดตข้อมูล"):
                    st.cache_data.clear(); st.rerun()
            except Exception as e:
                st.error(f"เชื่อมต่อล้มเหลว: {e}")
                st.info("💡 คำแนะนำ: โปรดตรวจสอบว่าเปิดสิทธิ์การแชร์เป็น 'ทุกคนที่มีลิงก์' แล้วหรือไม่")

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
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/13P9k7ucYHjbNQ88EucKXnR7JvPwGLEHF/edit?usp=drive_link" target="_blank">📄 1. พรรณนา/Daily Curve/Spot Map</a>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1kZSskpErufY_9qTl-_1TZaVymGMnNikm/edit?usp=drive_link" target="_blank">🕒 2. Hourly Epidemic Curve</a>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1TPJDOoIWCiZBtsnXDlhcHcN5IM27TBOK/edit?usp=drive_link" target="_blank">🔬 3. Case Control Analysis</a>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1HR57-mVqo9TceAgF1tpzWvLQi662akzw/edit?usp=drive_link" target="_blank">📊 4. Cohort Study Analysis</a>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. MAIN CONTENT
# ==========================================

if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    render_hero("ลงทะเบียนเข้าใช้งานระบบ", "ระบุหน่วยงานและวัตถุประสงค์ เพื่อเข้าสู่เมนูวิเคราะห์ข้อมูลระบาดวิทยา")
    section_header("📝", "ข้อมูลผู้ใช้งาน", "ใช้สำหรับแสดงบริบทของผู้วิเคราะห์ในระบบ")
    with st.form("registration"):
        u_agency = st.text_input("หน่วยงานต้นสังกัด (เช่น สสจ.อุดรธานี)")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรคภาคสนาม", "วิเคราะห์สถิติวิชาการ", "ซ้อมแผนฯ"])
        if st.form_submit_button("เริ่มใช้งาน"):
            if u_agency:
                st.session_state['registered'] = True
                st.success("ลงทะเบียนสำเร็จ!")
                st.rerun()
            else: st.error("กรุณาระบุหน่วยงาน")

elif df is not None:
    total_n = len(df)
    render_hero("Epi-Analytic Pro", "แพลตฟอร์มวิเคราะห์ข้อมูลระบาดวิทยาเชิงพรรณนา เชิงเวลา เชิงพื้นที่ และวิเคราะห์ปัจจัยเสี่ยง พร้อมผู้ช่วย AI สำหรับสรุปผล")
    render_data_overview(df)

    # ------------------------------------------
    # 6.1 Attack Rate
    # ------------------------------------------
    if menu == "👥 ประชากรและอัตราป่วย (Attack Rate)":
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

        if 'ar_context' in st.session_state:
            if st.button("✨ ให้ AI ช่วยสรุปผล", key="ai_ar"):
                with st.spinner("AI กำลังวิเคราะห์และสรุปผล..."):
                    summary = generate_ai_summary(api_key_input, st.session_state['ar_context'], "ประชากรและอัตราป่วย")
                    st.markdown(f"<div class='ai-summary-box'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)

    # ------------------------------------------
    # 6.2 Descriptive Analysis
    # ------------------------------------------
    elif menu == "👤 พรรณนา (Descriptive)":
        section_header("👤", "ระบาดวิทยาเชิงพรรณนา", "สรุปการกระจายตามบุคคล อายุ เพศ อาการ และสถิติเชิงปริมาณ")
        st.info(f"📋 จำนวนผู้ป่วยทั้งหมด (n) = {total_n} ราย")
        
        c1, c2 = st.columns(2)
        res_sex_str, res_age_str, s_df_str, numeric_stats_str = "", "", "", ""
        with c1:
            sex_col = st.selectbox("ตัวแปรเพศ", df.columns)
            res_sex = df[sex_col].value_counts().reset_index()
            res_sex.columns = ['เพศ', 'n']; res_sex['%'] = (res_sex['n']/total_n*100)
            st.table(res_sex.style.format({'%': '{:.2f}'}))
            res_sex_str = res_sex.to_string()
        with c2:
            age_col = st.selectbox("ตัวแปรอายุ", df.columns)
            df['age_grp'] = pd.cut(pd.to_numeric(df[age_col], errors='coerce'), bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
            res_age = df['age_grp'].value_counts().sort_index().reset_index()
            res_age.columns = ['อายุ', 'n']; res_age['%'] = (res_age['n']/total_n*100)
            st.table(res_age.style.format({'%': '{:.2f}'}))
            res_age_str = res_age.to_string()

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
            
            numeric_stats_str = f"\nสถิติเชิงปริมาณ ({num_col}): ค่าเฉลี่ย={mean_val:.2f}, มัธยฐาน={median_val:.2f}, SD={sd_val:.2f}, พิสัย(ต่ำสุด-สูงสุด)={min_val:.2f}-{max_val:.2f}"
        else:
            st.warning("⚠️ ข้อมูลที่เลือกไม่สามารถคำนวณค่าทางสถิติได้ (กรุณาเลือกคอลัมน์ที่เป็นตัวเลข)")

        st.markdown("---")
        st.subheader("อาการแสดง (1=มีอาการ)")
        symp_cols = st.multiselect("เลือกตัวแปรอาการ", df.columns)
        if symp_cols:
            s_df = pd.DataFrame([{"อาการ": c, "%": (df[c]==1).sum()/total_n*100} for c in symp_cols]).sort_values("%", ascending=True)
            s_df_str = s_df.to_string()
            
            fig_s = px.bar(s_df, x="%", y="อาการ", orientation='h', text_auto='.1f', color_discrete_sequence=['#E91E63'])
            fig_s.update_layout(font=dict(family="Sarabun", size=16, color="#4A4A4A"), title="แผนภูมิแท่งแนวนอนแสดงร้อยละของอาการ")
            
            st.plotly_chart(fig_s, use_container_width=True, config=high_res_config)
            st.caption("📸 คลิกที่ไอคอนกล้องถ่ายรูปมุมขวาบนของแผนภูมิแท่ง เพื่อดาวน์โหลดรูปภาพความละเอียดสูง")

        if st.button("✨ ให้ AI ช่วยสรุปผล", key="ai_desc"):
            with st.spinner("AI กำลังวิเคราะห์และสรุปผล..."):
                context = f"จำนวนเคส: {total_n}\nเพศ:\n{res_sex_str}\nอายุ:\n{res_age_str}{numeric_stats_str}\nอาการ:\n{s_df_str}"
                summary = generate_ai_summary(api_key_input, context, "ระบาดวิทยาเชิงพรรณนา")
                st.markdown(f"<div class='ai-summary-box'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)

    # ------------------------------------------
    # 6.3 Epidemic Curve 
    # ------------------------------------------
    elif menu == "📊 สร้าง Epi Curve (Time)":
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
        
        date_col = st.sidebar.selectbox("คอลัมน์วันเริ่มป่วย", df.columns)
        col_grp = st.sidebar.selectbox("ตัวแปรแยกกลุ่มสี:", ["<none>"] + list(df.columns))
        
        custom_color = st.sidebar.color_picker("🎨 เลือกสีแผนภูมิแท่งหลัก", "#E91E63")

        unit_map = {"Hour": "h", "Day": "d", "Week": "W", "Month": "ME", "30 Min": "30min"}
        bin_unit = st.sidebar.selectbox("หน่วยเวลา", list(unit_map.keys()), index=0)
        bin_size = st.sidebar.number_input("ขนาด Bin", min_value=1, value=1)
        freq = f"{bin_size}{unit_map[bin_unit]}"

        pad_before = st.sidebar.number_input(f"เพิ่มช่วงว่างก่อนหน้า ({bin_unit})", value=1)
        pad_after = st.sidebar.number_input(f"เพิ่มช่วงว่างข้างหลัง ({bin_unit})", value=1)

        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df_clean = df.dropna(subset=[date_col]).copy()

        if not df_clean.empty:
            min_dt, max_dt = df_clean[date_col].min(), df_clean[date_col].max()
            
            if "h" in freq or "min" in freq:
                start_range = (min_dt - pd.Timedelta(hours=pad_before)).floor('h')
                end_range = (max_dt + pd.Timedelta(hours=pad_after)).ceil('h')
            else:
                start_range = (min_dt - pd.to_timedelta(pad_before, unit='d')).floor('d')
                end_range = (max_dt + pd.to_timedelta(pad_after, unit='d')).ceil('d')
            
            full_range = pd.date_range(start=start_range, end=end_range, freq=freq)

            if col_grp == "<none>":
                counts = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size()
                chart_df = counts.reindex(full_range, fill_value=0).reset_index()
                chart_df.columns = [date_col, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', text_auto=True, color_discrete_sequence=[custom_color])
            else:
                counts = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().unstack(fill_value=0)
                chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name='Cases')
                chart_df.columns = [date_col, col_grp, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp, color_discrete_sequence=px.colors.sequential.RdPu[::-1])

            fig.update_layout(
                font=dict(family="Sarabun", size=16, color="#4A4A4A"),
                title="แผนภูมิแท่งแสดงการกระจายตัวของผู้ป่วยตามเวลาเริ่มป่วย (Epidemic Curve)",
                bargap=0.01, 
                xaxis=dict(type='date', tickformat='%d/%m %H:%M'),
                xaxis_title="Onset Date/Time",
                yaxis_title="Number of Cases",
                hovermode="x unified"
            )
            fig.update_traces(marker_line_width=0.5, marker_line_color='white')
            
            st.plotly_chart(fig, use_container_width=True, config=high_res_config)
            st.caption("📸 คลิกที่ไอคอนกล้องถ่ายรูปมุมขวาบนของแผนภูมิแท่ง เพื่อดาวน์โหลดรูปภาพความละเอียดสูง")

            if st.button("✨ ให้ AI ช่วยสรุปผล", key="ai_curve"):
                with st.spinner("AI กำลังวิเคราะห์และสรุปผล..."):
                    context = f"ตารางข้อมูลอนุกรมเวลา (Onset Date -> Cases):\n{chart_df.to_string()}"
                    summary = generate_ai_summary(api_key_input, context, "Epidemic Curve")
                    st.markdown(f"<div class='ai-summary-box'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)
        else:
            st.error("❌ ไม่สามารถวิเคราะห์ได้ เนื่องจากรูปแบบวันที่ในไฟล์ไม่ถูกต้อง")

    # ------------------------------------------
    # 6.4 Spot Map
    # ------------------------------------------
    elif menu == "🗺️ Spot Map (Place)":
        section_header("🗺️", "Spot Map - GIS Analytics", "แสดงตำแหน่งผู้ป่วย พื้นที่เสี่ยง และรัศมีควบคุมโรคบนแผนที่")
        lat_c = next((c for c in df.columns if any(p in c.lower() for p in ['lat', 'latitude', 'ละติจูด'])), None)
        lon_c = next((c for c in df.columns if any(p in c.lower() for p in ['lon', 'longitude', 'ลองจิจูด'])), None)
        
        if lat_c and lon_c:
            df_m = df.dropna(subset=[lat_c, lon_c]).copy()

            st.sidebar.markdown("---")
            st.sidebar.subheader("⚙️ ตั้งค่าแผนที่")
            
            info_cols = st.sidebar.multiselect(
                "เลือกข้อมูลที่จะโชว์บนป้าย Popup:",
                df.columns.tolist(),
                default=[df.columns[0]] if len(df.columns) > 0 else []
            )
            
            show_label_always = st.sidebar.checkbox("📌 โชว์ป้ายข้อมูลตลอดเวลา", value=True)
            
            buffer_radius = st.sidebar.number_input("รัศมีควบคุมโรค (เมตร)", min_value=0, value=100, step=50)
            map_type = st.sidebar.radio("รูปแบบแผนที่", ["ดาวเทียม (Google Hybrid)", "แผนที่ถนน (OpenStreetMap)"])

            if map_type == "ดาวเทียม (Google Hybrid)":
                tiles_url = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
                attr = 'Google'
            else:
                tiles_url = 'OpenStreetMap'
                attr = 'OpenStreetMap'

            m = folium.Map(
                location=[df_m[lat_c].mean(), df_m[lon_c].mean()], 
                zoom_start=16, 
                tiles=tiles_url, 
                attr=attr
            )

            for idx, r in df_m.iterrows():
                popup_content = f"<div style='font-family: Sarabun; font-size: 14px; white-space: nowrap;'>"
                for col in info_cols:
                    popup_content += f"<b>{col}:</b> {r[col]}<br>"
                popup_content += "</div>"
                
                if not info_cols: 
                    popup_content = f"<div style='font-family: Sarabun; font-size: 14px; white-space: nowrap;'>เคสที่ {idx+1}</div>"

                if buffer_radius > 0:
                    folium.Circle(
                        location=[r[lat_c], r[lon_c]], 
                        radius=buffer_radius, 
                        color='#FFEB3B', 
                        weight=2,
                        fill=True,
                        fill_opacity=0.25,
                        fill_color='#FF9800'
                    ).add_to(m)

                marker = folium.CircleMarker(
                    location=[r[lat_c], r[lon_c]], 
                    radius=6, 
                    color='#E91E63',
                    fill=True, 
                    fill_opacity=1.0
                )
                
                if show_label_always:
                    marker.add_child(folium.Tooltip(popup_content, permanent=True, direction='right', opacity=0.85))
                else:
                    marker.add_child(folium.Popup(popup_content, max_width=300))
                    
                marker.add_to(m)

            components.html(m._repr_html_(), height=650)
            st.caption("💡 แนะนำให้ใช้ฟังก์ชัน Screen Capture (Print Screen) ของคอมพิวเตอร์ เพื่อบันทึกภาพแผนที่")

            if st.button("✨ ให้ AI ช่วยสรุปผล", key="ai_map"):
                with st.spinner("AI กำลังวิเคราะห์และสรุปผล..."):
                    context = f"พบผู้ป่วยจำนวน {len(df_m)} ราย กระจายตัวอยู่ในพื้นที่ ค่าเฉลี่ยพิกัดละติจูด: {df_m[lat_c].mean():.4f}, ลองจิจูด: {df_m[lon_c].mean():.4f}"
                    summary = generate_ai_summary(api_key_input, context, "Spot Map")
                    st.markdown(f"<div class='ai-summary-box'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)
        else: 
            st.warning("⚠️ ไม่พบคอลัมน์พิกัด (Lat/Lon) ในไฟล์ กรุณาตรวจสอบชื่อคอลัมน์")

    # ------------------------------------------
    # 6.5 Bivariate Analysis
    # ------------------------------------------
    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        section_header("🔬", "Bivariate Analysis & 2x2 Table", "คำนวณ OR/RR, 95% CI และ Mid-P exact สำหรับปัจจัยเสี่ยง")

        tab1, tab2 = st.tabs(["📁 วิเคราะห์จากไฟล์ข้อมูล", "🔢 กรอกข้อมูลเอง (Manual 2x2)"])

        with tab1:
            st.subheader("📁 วิเคราะห์ปัจจัยเสี่ยงจากไฟล์ที่อัปโหลด")
            if df is not None:
                out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns, key="file_out")
                design = st.radio("ประเภทการศึกษา", ["Case-control Study (OR)", "Cohort Study (RR)"], key="file_design")
                exp_list = st.multiselect("เลือกปัจจัยเสี่ยง", [c for c in df.columns if c != out_v], key="file_exp")

                if st.button("🚀 ประมวลผลจากไฟล์"):
                    results = []
                    for exp_v in exp_list:
                        temp = df[[out_v, exp_v]].copy().dropna()
                        temp[out_v] = smart_map_variable(temp[out_v])
                        temp[exp_v] = smart_map_variable(temp[exp_v])
                        temp = temp[temp[out_v].isin([1, 0]) & temp[exp_v].isin([1, 0])]

                        if len(temp) > 0:
                            a = len(temp[(temp[exp_v]==1) & (temp[out_v]==1)])
                            b = len(temp[(temp[exp_v]==1) & (temp[out_v]==0)])
                            c = len(temp[(temp[exp_v]==0) & (temp[out_v]==1)])
                            d = len(temp[(temp[exp_v]==0) & (temp[out_v]==0)])

                            try:
                                if "Case-control" in design:
                                    m_label = "OR"
                                    measure = (a * d) / (b * c) if (b * c) > 0 else 0
                                    se_ln = math.sqrt(1/a + 1/b + 1/c + 1/d) if a*b*c*d > 0 else 0
                                else:
                                    m_label = "RR"
                                    measure = (a / (a + b)) / (c / (c + d)) if (a+b) > 0 and (c+d) > 0 else 0
                                    se_ln = math.sqrt((1/a - 1/(a+b)) + (1/c - 1/(c+d))) if a*c > 0 else 0

                                ci_l = math.exp(math.log(measure) - 1.96 * se_ln) if measure > 0 else 0
                                ci_u = math.exp(math.log(measure) + 1.96 * se_ln) if measure > 0 else 0

                                mid_p_val = calculate_mid_p(a, b, c, d)

                                results.append({
                                    "ปัจจัย": exp_v, 
                                    "ป่วย(+)": a, "ไม่ป่วย(+)": b, 
                                    "ป่วย(-)": c, "ไม่ป่วย(-)": d, 
                                    m_label: measure, 
                                    "95% CI Lower": ci_l, 
                                    "95% CI Upper": ci_u, 
                                    "Mid-P (2-tail)": max(mid_p_val, 0)
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
                    else:
                        st.warning("⚠️ ไม่พบข้อมูลที่เพียงพอในการวิเคราะห์")

            if 'biv_file_res' in st.session_state:
                if st.button("✨ ให้ AI ช่วยสรุปผล", key="ai_biv_file"):
                    with st.spinner("AI กำลังวิเคราะห์และสรุปผล..."):
                        summary = generate_ai_summary(api_key_input, st.session_state['biv_file_res'], "Bivariate Analysis (จากไฟล์)")
                        st.markdown(f"<div class='ai-summary-box'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)

        with tab2:
            st.subheader("🔢 Manual 2x2 Table Calculator")
            st.info("ใช้สำหรับคำนวณกรณีมีเพียงตัวเลขสรุป (Aggregated Data) โดยไม่ต้องอัปโหลดไฟล์")

            manual_design = st.radio(
                "รูปแบบการศึกษา (Study Design):",
                ["Cohort Study (Relative Risk)", "Case-Control Study (Odds Ratio)"],
                horizontal=True, key="man_design"
            )

            st.markdown("---")
            c1, c2, c3 = st.columns([2, 1, 1])

            with c1:
                st.write("") 
                st.write("")
                st.markdown("**Exposed (สัมผัสปัจจัย)**")
                st.write("")
                st.markdown("**Non-Exposed (ไม่สัมผัส)**")

            with c2:
                st.markdown("<center><b>Sick (ป่วย)</b></center>", unsafe_allow_html=True)
                ma = st.number_input("Cell a", min_value=0, value=0, step=1, label_visibility="collapsed")
                mc = st.number_input("Cell c", min_value=0, value=0, step=1, label_visibility="collapsed")

            with c3:
                st.markdown("<center><b>Not Sick (ไม่ป่วย)</b></center>", unsafe_allow_html=True)
                mb = st.number_input("Cell b", min_value=0, value=0, step=1, label_visibility="collapsed")
                md = st.number_input("Cell d", min_value=0, value=0, step=1, label_visibility="collapsed")

            if st.button("📈 คำนวณผล 2x2 Table"):
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
                        col_res1, col_res2 = st.columns(2)

                        with col_res1:
                            st.metric(res_label, f"{val:.2f}")
                            st.write(f"**95% CI (Taylor Series):**")
                            st.write(f"👉 {lower:.3f} - {upper:.3f}")
                            st.caption("ค่านี้จะตรงกับผลลัพธ์ใน OpenEpi/Epi Info")

                        with col_res2:
                            st.write("**Statistical Significance**")
                            st.write(f"**Yates chi-square:** {chi2_yates:.3f}")
                            st.write(f"**Mid-P exact (2-tail):** {max(mid_p_val, 0.0000001):.7f}")

                            if mid_p_val < 0.05:
                                st.success("✨ มีนัยสำคัญทางสถิติ (p < 0.05)")
                            else:
                                st.error("❌ ไม่มีนัยสำคัญทางสถิติ")

                        manual_res = f"Study Design: {manual_design}\n{res_label}: {val:.2f} (95% CI: {lower:.3f} - {upper:.3f})\nYates chi-square: {chi2_yates:.3f}\nMid-P exact: {max(mid_p_val, 0.0000001):.7f}"
                        st.session_state['biv_man_res'] = manual_res

                    except Exception as e:
                        st.error(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ: {e}")
                else:
                    st.warning("กรุณากรอกตัวเลขจำนวนในตาราง 2x2")

            if 'biv_man_res' in st.session_state:
                if st.button("✨ ให้ AI ช่วยสรุปผล", key="ai_biv_man"):
                    with st.spinner("AI กำลังวิเคราะห์และสรุปผล..."):
                        summary = generate_ai_summary(api_key_input, st.session_state['biv_man_res'], "Bivariate Analysis (Manual)")
                        st.markdown(f"<div class='ai-summary-box'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)

    # ------------------------------------------
    # 6.6 Logistic Regression
    # ------------------------------------------
    elif menu == "🧬 Multiple Logistic Regression (AOR)":
        section_header("🧬", "Multiple Logistic Regression", "วิเคราะห์ Adjusted OR โดยควบคุมตัวแปรกวน")
        out_v = st.selectbox("Outcome", df.columns, key="mlr_out")
        exp_v = st.selectbox("ปัจจัยหลัก", [c for c in df.columns if c != out_v])
        adj_v = st.multiselect("ตัวแปรกวน", [c for c in df.columns if c not in [out_v, exp_v]])
        
        if st.button("🚀 คำนวณ AOR"):
            try:
                df_m = df[[out_v, exp_v] + adj_v].copy().dropna()
                for c in df_m.columns: df_m[c] = smart_map_variable(df_m[c])
                
                formula = f"Q('{out_v}') ~ Q('{exp_v}')"
                if adj_v: formula += " + " + " + ".join([f"Q('{a}')" for a in adj_v])
                
                model = smf.logit(formula, data=df_m).fit(disp=0)
                
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
                st.dataframe(res_df.style.format({
                    "Adjusted OR (AOR)": "{:.2f}",
                    "95% CI Lower": "{:.2f}",
                    "95% CI Upper": "{:.2f}",
                    "P-value": "{:.4f}"
                }).apply(lambda x: ['background-color: #F8BBD0' if x['P-value'] < 0.05 else '' for _ in x], axis=1), 
                use_container_width=True)
                
                st.success("✅ คำนวณค่า Adjusted OR และ 95% CI สำเร็จ")
                st.session_state['mlr_res'] = res_df.to_string()

            except Exception as e:
                st.error(f"⚠️ ไม่สามารถประมวลผลได้: {e}")

        if 'mlr_res' in st.session_state:
            if st.button("✨ ให้ AI ช่วยสรุปผล", key="ai_mlr"):
                with st.spinner("AI กำลังวิเคราะห์และสรุปผล..."):
                    summary = generate_ai_summary(api_key_input, st.session_state['mlr_res'], "Multiple Logistic Regression (AOR)")
                    st.markdown(f"<div class='ai-summary-box'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #880E4F;'>Epi-Analytic Pro ODPC8 | พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี กรมควบคุมโรค</div>", unsafe_allow_html=True)
