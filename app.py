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
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Epi-Analytic Pro ODPC8", 
    page_icon="🦠", 
    layout="wide"
)

# เพิ่ม CSS เพื่อปรับแต่ง Sidebar กลับเป็นโทนสีเทาเดิม
st.markdown(
    """
    <style>
        /* 1. ปรับสีพื้นหลัง Sidebar เป็นสีเทาจางๆ */
        [data-testid="stSidebar"] {
            background-color: #F8F9FB !important; /* สีเทาอ่อนสะอาดตา */
            border-right: 1px solid #E0E0E0;
        }

        /* 2. ปรับสีตัวอักษร หัวข้อ และ Label เป็นสีดำ/เทาเข้ม */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] .st-at {
            color: #31333F !important; /* สีดำมาตรฐาน Streamlit */
            font-weight: 500;
        }

        /* 3. ปรับสีปุ่ม Radio (ตัวเลือกเมนู) ให้เป็นสีดำ */
        [data-testid="stSidebar"] .st-bc, 
        [data-testid="stSidebar"] .st-bd {
            color: #31333F !important;
        }

        /* 4. ปรับแต่งให้โลโก้ดูคมชัด */
        [data-testid="stSidebar"] img {
            object-fit: contain;
            image-rendering: -webkit-optimize-contrast;
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
    st.sidebar.title("🏥 ODPC8 Udon Thani")

st.sidebar.markdown("---")

st.sidebar.title("🏥 Epi-Analytic Menu")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนเพื่อปลดล็อกเมนูวิเคราะห์\n\n🛡️ ความปลอดภัย: โปรดตรวจสอบไฟล์และลบข้อมูลระบุตัวตนออกก่อนอัปโหลด")
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
    st.sidebar.subheader("💾 แหล่งข้อมูล (Data Source)")
    data_source = st.sidebar.radio("เลือกแหล่งข้อมูล:", ["อัปโหลดไฟล์ (CSV/Excel)", "Google Sheets (Real-time)"])

    if data_source == "อัปโหลดไฟล์ (CSV/Excel)":
        uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ข้อมูล", type=['xlsx', 'csv'])
        if uploaded_file:
            df = load_data(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input("🔗 วางลิงก์ Google Sheets ที่นี่:")
        if sheet_url:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(spreadsheet=sheet_url)
                if st.sidebar.button("🔄 อัปเดตข้อมูล (Refresh)"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"ไม่สามารถเชื่อมต่อ Google Sheets ได้: {e}")

# --- หน้าลงทะเบียน ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
    st.caption("ระบบบันทึกข้อมูลตามมาตรฐาน PDPA ไม่มีการเก็บชื่อ-นามสกุลของผู้ใช้งาน")

    with st.form("reg_form_v2"):
        u_team = st.selectbox("ประเภททีม", ["CDCU", "SRRT", "SAT", "JIT", "อื่นๆ"])
        u_agency = st.text_input("หน่วยงาน / สังกัด (เช่น สสจ.อุดรธานี, รพ.เลย)")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรคภาคสนาม", "วิเคราะห์สถิติวิชาการ", "ซ้อมแผนฯ", "อื่นๆ"])
        
        submit_reg = st.form_submit_button("เริ่มใช้งานระบบ")

        if submit_reg:
            if not u_agency:
                st.error("กรุณาระบุหน่วยงานก่อนเข้าใช้งาน")
            else:
                from datetime import datetime
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                payload = {"timestamp": now, "team": u_team, "agency": u_agency, "purpose": u_purpose}
                try:
                    url = "https://script.google.com/macros/s/AKfycbxVGzrB9IjdvD90g2Zm8cKNwYE1PMrtaaun7YlBkGjWoL3UjVw74K49B_wg4cBfedeB/exec"
                    response = requests.post(url, json=payload)
                    if response.status_code == 200:
                        st.session_state['registered'] = True
                        st.success("✅ บันทึกประวัติการเข้าใช้งานเรียบร้อย")
                        st.balloons()
                        st.rerun()
                    else:
                        raise Exception("เซิร์ฟเวอร์ตอบกลับด้วยสถานะอื่น")
                except Exception as e:
                    st.session_state['registered'] = True 
                    st.warning(f"⚠️ บันทึกสถิติไม่สำเร็จ แต่ท่านสามารถใช้งานแอปได้ปกติ")

# --- ส่วนการวิเคราะห์ ---
elif st.session_state['registered'] and df is not None:
    total_n = len(df)

    if menu == "👥 ประชากรและอัตราป่วย (Attack Rate)":
        st.title("👥 ประชากรและอัตราป่วย (Attack Rate)")
        st.info("กรุณาระบุจำนวนประชากรกลุ่มเสี่ยง (Population at Risk) เพื่อคำนวณอัตราป่วย")

        def find_col(possible_names):
            return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)
        
        sex_c = find_col(['sex', 'gender', 'เพศ'])
        age_c = find_col(['age', 'อายุ'])

        st.subheader("1. ระบุจำนวนประชากรกลุ่มเสี่ยงแยกตามกลุ่ม")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("**แยกตามเพศ**")
            pop_male = st.number_input("จำนวนประชากรชายทั้งหมด", min_value=1, value=100)
            pop_female = st.number_input("จำนวนประชากรหญิงทั้งหมด", min_value=1, value=100)

        with col_p2:
            st.markdown("**แยกตามกลุ่มอายุ (มาตรฐาน)**")
            age_labels = ['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+']
            pop_age = {label: st.number_input(f"ประชากรกลุ่มอายุ {label}", min_value=0, value=0) for label in age_labels}

        if st.button("📈 คำนวณ Attack Rate"):
            st.markdown("---")
            total_pop = pop_male + pop_female
            overall_ar = (len(df) / total_pop * 100) if total_pop > 0 else 0
            st.metric("Overall Attack Rate", f"{overall_ar:.2f} %", f"Cases: {len(df)}")

            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("**Sex-Specific Attack Rate**")
                if sex_c:
                    df['sex_temp'] = df[sex_c].astype(str).str.strip().replace({'1':'ชาย','2':'หญิง','1.0':'ชาย','2.0':'หญิง'})
                    male_cases = len(df[df['sex_temp'] == 'ชาย'])
                    female_cases = len(df[df['sex_temp'] == 'หญิง'])
                    ar_sex_df = pd.DataFrame({
                        "เพศ": ["ชาย", "หญิง"],
                        "จำนวนป่วย (n)": [male_cases, female_cases],
                        "ประชากร (N)": [pop_male, pop_female],
                        "Attack Rate (%)": [male_cases/pop_male*100, female_cases/pop_female*100]
                    })
                    st.table(ar_sex_df.style.format({"Attack Rate (%)": "{:.2f}"}))
                
            with res_col2:
                st.markdown("**Age-Specific Attack Rate**")
                if age_c:
                    df['age_grp_temp'] = pd.cut(df[age_c], bins=[0,5,15,25,35,45,55,65,120], labels=age_labels, right=False)
                    age_cases = df['age_grp_temp'].value_counts().reindex(age_labels, fill_value=0)
                    age_ar_data = [{"กลุ่มอายุ": label, "จำนวนป่วย (n)": age_cases[label], "ประชากร (N)": pop_age[label], "Attack Rate (%)": (age_cases[label]/pop_age[label]*100) if pop_age[label]>0 else 0} for label in age_labels]
                    st.table(pd.DataFrame(age_ar_data).style.format({"Attack Rate (%)": "{:.2f}"}))

    elif menu == "👤 บุคคล (Person)":
        st.title("👤 การกระจายตามบุคคล")
        st.info(f"📋 จำนวนข้อมูลทั้งหมด (n) = {total_n} ราย")
        
        col1, col2 = st.columns(2)
        with col1:
            sel_sex = st.selectbox("เลือกตัวแปรเพศ", df.columns)
            res_sex = df[sel_sex].value_counts().reset_index()
            res_sex.columns = ['เพศ', 'จำนวน (n)']
            res_sex['ร้อยละ (%)'] = (res_sex['จำนวน (n)']/total_n*100).round(2)
            st.table(res_sex)
            
        with col2:
            sel_age = st.selectbox("เลือกตัวแปรอายุ", df.columns)
            df['age_group'] = pd.cut(df[sel_age], bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
            res_age = df['age_group'].value_counts().sort_index().reset_index()
            res_age.columns = ['กลุ่มอายุ', 'จำนวน (n)']
            res_age['ร้อยละ (%)'] = (res_age['จำนวน (n)']/total_n*100).round(2)
            st.table(res_age)

    elif menu == "📊 Epidemic Curve (Time)":
        st.title("📊 Epidemic Curve")
        date_col = st.selectbox("เลือกตัวแปรเวลา (Onset Date/Time)", df.columns)
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df_clean = df.dropna(subset=[date_col])
        
        freq = st.radio("เลือกความละเอียด:", ["H (รายชั่วโมง)", "D (รายวัน)"], horizontal=True)
        f_code = freq[0]
        
        counts = df_clean.groupby(pd.Grouper(key=date_col, freq=f_code)).size().reset_index(name='Cases')
        fig = px.bar(counts, x=date_col, y='Cases', color_discrete_sequence=['#3498db'], text_auto=True)
        fig.update_layout(xaxis_title="เวลาที่เริ่มป่วย", yaxis_title="จำนวนราย", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        st.title("🔬 Bivariate Analysis (OR/RR)")
        tab1, tab2 = st.tabs(["📁 วิเคราะห์จากไฟล์ข้อมูล", "🔢 กรอกข้อมูลเอง (Manual 2x2)"])
        
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

        with tab2:
            st.subheader("🔢 Manual 2x2 Table")
            ma = st.number_input("Cell a (Exposed Sick)", min_value=0, value=0)
            mb = st.number_input("Cell b (Exposed Not Sick)", min_value=0, value=0)
            mc = st.number_input("Cell c (Non-Exposed Sick)", min_value=0, value=0)
            md = st.number_input("Cell d (Non-Exposed Not Sick)", min_value=0, value=0)
            
            if st.button("📈 คำนวณ"):
                or_val = (ma*md)/(mb*mc) if (mb*mc)>0 else 0
                mid_p = calculate_mid_p(ma, mb, mc, md)
                st.metric("Odds Ratio (OR)", f"{or_val:.2f}")
                st.write(f"Mid-P Exact: {mid_p:.4f}")

    elif menu == "🧬 Multiple Logistic Regression (AOR)":
        st.title("🧬 Multiple Logistic Regression")
        out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns, key="log_out")
        exp_v = st.selectbox("ปัจจัยหลัก (Exposure)", [c for c in df.columns if c != out_v], key="log_exp")
        adj_v = st.multiselect("ตัวแปรกวน (Confounding)", [c for c in df.columns if c not in [out_v, exp_v]], key="log_adj")
        
        if st.button("🚀 ประมวลผล AOR"):
            try:
                cols = [out_v, exp_v] + adj_v
                df_m = df[cols].copy().dropna()
                for col in df_m.columns:
                    df_m[col] = smart_map_variable(df_m[col])
                
                formula = f"Q('{out_v}') ~ Q('{exp_v}')"
                if adj_v: formula += " + " + " + ".join([f"Q('{a}')" for a in adj_v])
                
                model = smf.logit(formula, data=df_m).fit(disp=0)
                res_df = pd.DataFrame({
                    "Factors": model.params.index,
                    "Adjusted OR (AOR)": np.exp(model.params.values),
                    "P-value": model.pvalues.values
                })
                st.table(res_df[res_df['Factors'] != 'Intercept'].style.format({"Adjusted OR (AOR)": "{:.2f}", "P-value": "{:.4f}"}))
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map")
        lat_c = next((c for c in df.columns if any(p in c.lower() for p in ['lat', 'latitude'])), None)
        lon_c = next((c for c in df.columns if any(p in c.lower() for p in ['lon', 'longitude'])), None)
        
        if lat_c and lon_c:
            df_m = df.dropna(subset=[lat_c, lon_c]).copy()
            m = folium.Map(location=[df_m[lat_c].mean(), df_m[lon_c].mean()], zoom_start=15)
            for _, r in df_m.iterrows():
                folium.CircleMarker([r[lat_c], r[lon_c]], radius=7, color='#e74c3c', fill=True, fill_opacity=0.7).add_to(m)
            folium_static(m, width=1000)
        else:
            st.error("ไม่พบคอลัมน์พิกัดในไฟล์")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)
