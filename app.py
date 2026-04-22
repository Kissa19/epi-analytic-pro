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
# 1. CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="Epi-Analytic Pro ODPC8", 
    page_icon="🦠", 
    layout="wide"
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #F8F9FB !important;
            border-right: 1px solid #E0E0E0;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #31333F !important;
            font-weight: 500;
        }
        .stButton > button {
            border-radius: 8px;
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
# 3. HELPER FUNCTIONS
# ==========================================
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
    if unique_vals.issubset({1, 2, 1.0, 2.0, '1', '2'}):
        return pd.to_numeric(series, errors='coerce').map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    return series

def calculate_mid_p(a, b, c, d):
    n = a + b + c + d
    if n == 0: return np.nan
    k = a + c 
    m = a + b 
    p_obs = hypergeom.pmf(a, n, k, m)
    p_lower = hypergeom.cdf(a, n, k, m)
    p_upper = hypergeom.sf(a-1, n, k, m)
    mid_p = 2 * (min(p_lower, p_upper) - 0.5 * p_obs)
    return max(mid_p, 0.0)

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
try:
    st.sidebar.image("odpc8_logo.png", use_container_width=True)
except:
    st.sidebar.title("🏥 ODPC8 Udon Thani")

st.sidebar.markdown("---")
st.sidebar.title("🏥 Epi-Analytic Menu")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนก่อนเข้าใช้งาน")
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

# ==========================================
# 5. DATA LOADING
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
        sheet_url = st.sidebar.text_input("🔗 วางลิงก์ Google Sheets:")
        if sheet_url:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(spreadsheet=sheet_url)
                if st.sidebar.button("🔄 อัปเดตข้อมูล"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")

# ==========================================
# 6. ANALYTICS MODULES
# ==========================================

# --- หน้าลงทะเบียน ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
    with st.form("reg_form_v2"):
        u_team = st.selectbox("ประเภททีม", ["CDCU", "SRRT", "SAT", "JIT", "อื่นๆ"])
        u_agency = st.text_input("หน่วยงาน / สังกัด")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรคภาคสนาม", "วิเคราะห์สถิติวิชาการ", "ซ้อมแผนฯ", "อื่นๆ"])
        submit_reg = st.form_submit_button("เริ่มใช้งานระบบ")
        if submit_reg:
            if u_agency:
                st.session_state['registered'] = True
                st.success("บันทึกข้อมูลเรียบร้อย")
                st.rerun()
            else:
                st.error("กรุณาระบุหน่วยงาน")

# --- วิเคราะห์ Epidemic Curve (เวอร์ชัน Bin & Padding สมบูรณ์) ---
elif menu == "📊 Epidemic Curve (Time)" and df is not None:
    st.title("📊 Interactive Epidemic Curve")
    
    st.sidebar.subheader("⚙️ ตั้งค่าแกนเวลา")
    date_col = st.sidebar.selectbox("เลือกตัวแปรเวลา (Onset Date/Time)", df.columns)
    col_grp = st.sidebar.selectbox("ตัวแปรแยกกลุ่มสี:", ["<ไม่มี>"] + df.columns.tolist())
    
    unit_map = {"Hour": "H", "Day": "D", "Week": "W", "Month": "M"}
    bin_size = st.sidebar.number_input("ขนาด Bin (เช่น ทุก 2 ชม.)", min_value=1, value=1)
    bin_unit = st.sidebar.selectbox("หน่วยเวลา", list(unit_map.keys()), index=0)
    freq = f"{bin_size}{unit_map[bin_unit]}"

    pad_before = st.sidebar.number_input(f"เพิ่มช่วงก่อนหน้า ({bin_unit})", value=2)
    pad_after = st.sidebar.number_input(f"เพิ่มช่วงข้างหลัง ({bin_unit})", value=2)

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df_clean = df.dropna(subset=[date_col]).copy()

    if not df_clean.empty:
        min_dt = df_clean[date_col].min()
        max_dt = df_clean[date_col].max()

        # คำนวณช่วงเวลาแบบมี Padding
        if bin_unit == "Hour":
            start_range = (min_dt - pd.Timedelta(hours=pad_before * bin_size)).floor('H')
            end_range = (max_dt + pd.Timedelta(hours=pad_after * bin_size)).ceil('H')
        else:
            offset = pd.to_timedelta(pad_before * bin_size, unit=bin_unit[0].lower()) if bin_unit != "Month" else pd.DateOffset(months=pad_before)
            start_range = (min_dt - offset).floor('D')
            end_range = (max_dt + offset).ceil('D')

        full_range = pd.date_range(start=start_range, end=end_range, freq=freq)

        if col_grp == "<ไม่มี>":
            counts = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size()
            chart_df = counts.reindex(full_range, fill_value=0).reset_index()
            chart_df.columns = [date_col, 'Cases']
            fig = px.bar(chart_df, x=date_col, y='Cases', color_discrete_sequence=["#3498db"], text_auto=True)
        else:
            counts = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().unstack(fill_value=0)
            chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name='Cases')
            chart_df.columns = [date_col, col_grp, 'Cases']
            fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp, color_discrete_sequence=px.colors.qualitative.Set1)

        fig.update_layout(bargap=0.05, xaxis_title="Onset Time", yaxis_title="Cases", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"แสดงผลราย {bin_size} {bin_unit} เรียบร้อยแล้ว")

# --- วิเคราะห์ Bivariate (OR/RR) ---
elif menu == "🔬 Bivariate Analysis (OR/RR)" and df is not None:
    st.title("🔬 Bivariate Analysis (OR/RR)")
    out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns)
    exp_list = st.multiselect("เลือกปัจจัยเสี่ยง (Exposures)", [c for c in df.columns if c != out_v])
    
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
            mid_p = calculate_mid_p(a, b, c, d)
            results.append({"Factor": exp_v, "a":a, "b":b, "c":c, "d":d, "Odds Ratio": or_val, "Mid-P": mid_p})
        
        st.dataframe(pd.DataFrame(results).style.format({"Odds Ratio": "{:.2f}", "Mid-P": "{:.4f}"}))

# --- วิเคราะห์ Multiple Logistic Regression (AOR) ---
elif menu == "🧬 Multiple Logistic Regression (AOR)" and df is not None:
    st.title("🧬 Multiple Logistic Regression (AOR)")
    out_v = st.selectbox("ตัวแปรตาม", df.columns)
    exp_v = st.selectbox("ปัจจัยหลัก", [c for c in df.columns if c != out_v])
    adj_v = st.multiselect("ตัวแปรกวน", [c for c in df.columns if c not in [out_v, exp_v]])
    
    if st.button("🚀 คำนวณ AOR"):
        try:
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
            st.table(res_df[res_df['Factors'] != 'Intercept'].style.format({"AOR": "{:.2f}", "P-value": "{:.4f}"}))
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)
