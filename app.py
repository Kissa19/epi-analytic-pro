import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import timedelta
import scipy.stats as stats
from scipy.stats import hypergeom
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import folium
from streamlit_folium import folium_static
import requests

# ==========================================
# 1. CONFIGURATION & THEME (DDC PINK)
# ==========================================
st.set_page_config(
    page_title="Epi-Analytic Pro ODPC8", 
    page_icon="🦠", 
    layout="wide"
)

# การฉีด CSS เพื่อปรับ Font (Kanit) และโทนสีชมพู กรมควบคุมโรค
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        /* 1. Global Font */
        html, body, [class*="css"], .stMarkdown, p, label, button, select, input {
            font-family: 'Kanit', sans-serif !important;
        }

        /* 2. Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 2px solid #F0F2F6;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #333333 !important;
            font-weight: 500;
        }

        /* 3. DDC Pink Accents */
        h1, h2, h3 {
            color: #E91E63 !important; /* สีชมพูหลักกรมควบคุมโรค */
            font-weight: 700 !important;
        }
        
        /* ปรับสีปุ่ม */
        .stButton > button {
            background-color: #E91E63 !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #C2185B !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* ปรับแต่ง Radio Button */
        div[data-testid="stMarkdownContainer"] > p > strong {
            color: #E91E63 !important;
        }
        
        /* Metric Box */
        [data-testid="stMetricValue"] {
            color: #E91E63 !important;
        }

        /* 4. Logo Styling */
        [data-testid="stSidebar"] img {
            object-fit: contain;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'registered' not in st.session_state:
    st.session_state['registered'] = False

# ==========================================
# 3. SIDEBAR NAVIGATION & LOGO
# ==========================================
try:
    st.sidebar.image("odpc8_logo.png", use_container_width=True)
except:
    st.sidebar.title("💗 ODPC8 Udon Thani")

st.sidebar.markdown("---")

st.sidebar.title("📊 ระบบวิเคราะห์ระบาดวิทยา")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนเพื่อปลดล็อกเมนูวิเคราะห์\n\n🛡️ ความปลอดภัย: โปรดลบชื่อ-นามสกุลออกก่อนอัปโหลดข้อมูล")
else:
    menu = st.sidebar.radio(
        "เลือกหัวข้อการวิเคราะห์", 
        ["👥 ประชากรและอัตราป่วย (Attack Rate)",
         "👤 บุคคล (Person)", 
         "📊 Epidemic Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Bivariate Analysis (OR/RR)", 
         "🧬 Multiple Logistic Regression (AOR)",
         "📝 ข้อมูลการลงทะเบียน (แก้ไข)"]
    )

# --- Helper Functions ---
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์ได้: {e}")
        return None

def smart_map_variable(series):
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({1, 2, 1.0, 2.0}):
        return series.map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    return series

def calculate_mid_p(a, b, c, d):
    n = a + b + c + d
    if n == 0: return np.nan
    k = a + c 
    m = a + b 
    p_obs = hypergeom.pmf(a, n, k, m)
    p_lower = hypergeom.cdf(a - 1, n, k, m) + 0.5 * p_obs
    p_upper = (1 - hypergeom.cdf(a, n, k, m)) + 0.5 * p_obs
    mid_p = 2 * min(p_lower, p_upper)
    return min(mid_p, 1.0)

# ==========================================
# 4. MAIN CONTENT AREA
# ==========================================

df = None
if st.session_state['registered']:
    st.sidebar.divider()
    st.sidebar.subheader("💾 แหล่งข้อมูล")
    data_source = st.sidebar.radio("เลือกแหล่งข้อมูล:", ["อัปโหลดไฟล์", "Google Sheets (Real-time)"])

    if data_source == "อัปโหลดไฟล์":
        uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ข้อมูล", type=['xlsx', 'csv'])
        if uploaded_file:
            df = load_data(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input("🔗 วางลิงก์ Google Sheets ที่นี่:")
        if sheet_url:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(spreadsheet=sheet_url)
                if st.sidebar.button("🔄 Refresh Data"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"ไม่สามารถเชื่อมต่อ Google Sheets ได้: {e}")

# --- หน้าลงทะเบียน ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
    st.info("เพื่อให้สอดคล้องกับมาตรฐาน PDPA ระบบจะไม่จัดเก็บชื่อจริงของท่าน")

    with st.form("reg_form_v2"):
        u_team = st.selectbox("ประเภททีม", ["CDCU", "SRRT", "SAT", "JIT", "อื่นๆ"])
        u_agency = st.text_input("หน่วยงาน / สังกัด (เช่น สสจ.อุดรธานี)")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรคภาคสนาม", "วิเคราะห์สถิติวิชาการ", "ซ้อมแผนฯ", "อื่นๆ"])
        
        submit_reg = st.form_submit_button("เข้าสู่ระบบวิเคราะห์")

        if submit_reg:
            if not u_agency:
                st.error("กรุณาระบุหน่วยงาน")
            else:
                from datetime import datetime
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                payload = {"timestamp": now, "team": u_team, "agency": u_agency, "purpose": u_purpose}
                try:
                    url = "https://script.google.com/macros/s/AKfycbxVGzrB9IjdvD90g2Zm8cKNwYE1PMrtaaun7YlBkGjWoL3UjVw74K49B_wg4cBfedeB/exec"
                    response = requests.post(url, json=payload)
                    st.session_state['registered'] = True
                    st.success("✅ บันทึกประวัติสำเร็จ")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.session_state['registered'] = True 
                    st.warning(f"⚠️ บันทึกสถิติไม่สำเร็จ แต่ท่านสามารถใช้งานได้ปกติ")

# --- ส่วนการวิเคราะห์ ---
elif st.session_state['registered'] and df is not None:
    total_n = len(df)

    if menu == "👥 ประชากรและอัตราป่วย (Attack Rate)":
        st.title("👥 Attack Rate Analysis")
        st.write("คำนวณอัตราป่วยแยกตามเพศและกลุ่มอายุ")

        def find_col(possible_names):
            return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)
        
        sex_c = find_col(['sex', 'gender', 'เพศ'])
        age_c = find_col(['age', 'อายุ'])

        st.subheader("📊 ระบุประชากรกลุ่มเสี่ยง (Denominator)")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("### แยกตามเพศ")
            pop_male = st.number_input("ประชากรชาย (N)", min_value=1, value=100)
            pop_female = st.number_input("ประชากรหญิง (N)", min_value=1, value=100)

        with col_p2:
            st.markdown("### แยกตามกลุ่มอายุ")
            age_labels = ['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+']
            pop_age = {label: st.number_input(f"อายุ {label}", min_value=0, value=0) for label in age_labels}

        if st.button("📈 เริ่มคำนวณ Attack Rate"):
            st.divider()
            overall_ar = (len(df) / (pop_male + pop_female) * 100)
            st.metric("Overall Attack Rate", f"{overall_ar:.2f} %", f"Total Cases: {len(df)}")

            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("**Sex-Specific AR**")
                if sex_c:
                    df['sex_temp'] = df[sex_c].astype(str).str.strip().replace({'1':'ชาย','2':'หญิง','1.0':'ชาย','2.0':'หญิง'})
                    male_cases = len(df[df['sex_temp'] == 'ชาย'])
                    female_cases = len(df[df['sex_temp'] == 'หญิง'])
                    ar_sex_df = pd.DataFrame({
                        "เพศ": ["ชาย", "หญิง"],
                        "ป่วย (n)": [male_cases, female_cases],
                        "ประชากร (N)": [pop_male, pop_female],
                        "AR (%)": [male_cases/pop_male*100, female_cases/pop_female*100]
                    })
                    st.table(ar_sex_df.style.format({"AR (%)": "{:.2f}"}))
                
            with res_col2:
                st.markdown("**Age-Specific AR**")
                if age_c:
                    df['age_grp_temp'] = pd.cut(df[age_c], bins=[0,5,15,25,35,45,55,65,120], labels=age_labels, right=False)
                    age_cases = df['age_grp_temp'].value_counts().reindex(age_labels, fill_value=0)
                    age_ar_data = [{"กลุ่มอายุ": label, "ป่วย (n)": age_cases[label], "ประชากร (N)": pop_age[label], "AR (%)": (age_cases[label]/pop_age[label]*100) if pop_age[label]>0 else 0} for label in age_labels]
                    st.table(pd.DataFrame(age_ar_data).style.format({"AR (%)": "{:.2f}"}))

    elif menu == "👤 บุคคล (Person)":
        st.title("👤 Descriptive Analysis: Person")
        
        col1, col2 = st.columns(2)
        with col1:
            sel_sex = st.selectbox("เลือกตัวแปรเพศ", df.columns)
            res_sex = df[sel_sex].value_counts().reset_index()
            res_sex.columns = ['เพศ', 'n']
            res_sex['%'] = (res_sex['n']/total_n*100).round(2)
            st.table(res_sex)
            
        with col2:
            sel_age = st.selectbox("เลือกตัวแปรอายุ", df.columns)
            df['age_group'] = pd.cut(df[sel_age], bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
            res_age = df['age_group'].value_counts().sort_index().reset_index()
            res_age.columns = ['กลุ่มอายุ', 'n']
            res_age['%'] = (res_age['n']/total_n*100).round(2)
            st.table(res_age)

    elif menu == "📊 Epidemic Curve (Time)":
        st.title("📊 Epidemic Curve")
        date_col = st.selectbox("ตัวแปรวันเริ่มป่วย", df.columns)
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df_clean = df.dropna(subset=[date_col])
        
        freq = st.radio("ความละเอียด", ["H", "D"], horizontal=True)
        counts = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name='Cases')
        
        fig = px.bar(counts, x=date_col, y='Cases', color_discrete_sequence=['#E91E63'], text_auto=True)
        fig.update_layout(title="Epidemic Curve", xaxis_title="Time", yaxis_title="Cases", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        st.title("🔬 Bivariate Analysis (OR/RR)")
        tab1, tab2 = st.tabs(["วิเคราะห์จากไฟล์", "กรอกข้อมูลเอง (Manual)"])
        
        with tab1:
            out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns)
            exp_list = st.multiselect("เลือกปัจจัยเสี่ยง", [c for c in df.columns if c != out_v])
            
            if st.button("🚀 ประมวลผล OR/RR"):
                results = []
                for exp_v in exp_list:
                    temp = df[[out_v, exp_v]].copy().dropna()
                    temp[out_v] = smart_map_variable(temp[out_v])
                    temp[exp_v] = smart_map_variable(temp[exp_v])
                    a = len(temp[(temp[exp_v]==1) & (temp[out_v]==1)])
                    b = len(temp[(temp[exp_v]==1) & (temp[out_v]==0)])
                    c = len(temp[(temp[exp_v]==0) & (temp[out_v]==1)])
                    d = len(temp[(temp[exp_v]==0) & (temp[out_v]==0)])
                    
                    or_val = (a*d)/(b*c) if (b*c)>0 else 0
                    mid_p = calculate_mid_p(a,b,c,d)
                    results.append({"ปัจจัย": exp_v, "OR": or_val, "Mid-P": mid_p})
                st.table(pd.DataFrame(results).style.format({"OR": "{:.2f}", "Mid-P": "{:.4f}"}))

    elif menu == "🧬 Multiple Logistic Regression (AOR)":
        st.title("🧬 Multiple Logistic Regression")
        out_v = st.selectbox("Outcome (ป่วย=1, ไม่ป่วย=0)", df.columns, key="mlr_out")
        exp_v = st.selectbox("ปัจจัยหลัก", [c for c in df.columns if c != out_v], key="mlr_exp")
        adj_v = st.multiselect("ตัวแปรกวน", [c for c in df.columns if c not in [out_v, exp_v]])
        
        if st.button("🚀 คำนวณ Adjusted OR"):
            cols = [out_v, exp_v] + adj_v
            df_m = df[cols].copy().dropna()
            for c in df_m.columns: df_m[c] = smart_map_variable(df_m[c])
            
            formula = f"Q('{out_v}') ~ Q('{exp_v}')"
            if adj_v: formula += " + " + " + ".join([f"Q('{a}')" for a in adj_v])
            
            model = smf.logit(formula, data=df_m).fit(disp=0)
            res_df = pd.DataFrame({
                "Factors": model.params.index,
                "AOR": np.exp(model.params.values),
                "P-value": model.pvalues.values
            })
            st.table(res_df[res_df['Factors'] != 'Intercept'])

    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map")
        lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
        lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
        
        if lat_col and lon_col:
            df_m = df.dropna(subset=[lat_col, lon_col])
            m = folium.Map(location=[df_m[lat_col].mean(), df_m[lon_col].mean()], zoom_start=15)
            for _, r in df_m.iterrows():
                folium.CircleMarker([r[lat_col], r[lon_col]], radius=7, color='#E91E63', fill=True).add_to(m)
            folium_static(m, width=1000)
        else:
            st.error("ไม่พบคอลัมน์พิกัด")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Epi-Analytic Pro | กรมควบคุมโรค (DDC) โทนสีชมพู อัตลักษณ์องค์กร</div>", unsafe_allow_html=True)
