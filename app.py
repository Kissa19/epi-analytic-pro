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
# 1. CONFIGURATION & PROFESSIONAL STYLING
# ==========================================
st.set_page_config(
    page_title="Epi-Analytic Pro | ODPC8", 
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
        .stButton > button { border-radius: 8px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. SESSION STATE & HELPERS
# ==========================================
if 'registered' not in st.session_state:
    st.session_state['registered'] = False

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'): return pd.read_csv(file)
        else: return pd.read_excel(file)
    except Exception as e:
        st.error(f"โหลดไฟล์ไม่สำเร็จ: {e}")
        return None

def smart_map_variable(series):
    """จัดการรหัส 1/2 ให้เป็น 1/0 สำหรับการคำนวณทางระบาดวิทยา"""
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({1, 2, 1.0, 2.0, '1', '2'}):
        return pd.to_numeric(series, errors='coerce').map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    return series

def calculate_mid_p(a, b, c, d):
    """คำนวณ Mid-P Exact P-value (2-tail) มาตรฐานสากล"""
    n = a + b + c + d
    if n == 0: return 1.0
    k, m = a + c, a + b
    p_obs = hypergeom.pmf(a, n, k, m)
    p_lower = hypergeom.cdf(a, n, k, m)
    p_upper = hypergeom.sf(a-1, n, k, m)
    mid_p = 2 * (min(p_lower, p_upper) - 0.5 * p_obs)
    return max(min(mid_p, 1.0), 0.0)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
try:
    st.sidebar.image("odpc8_logo.png", use_container_width=True)
except:
    st.sidebar.title("🏥 ODPC8 Udon Thani")

st.sidebar.markdown("---")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนก่อนเข้าสู่โหมดวิเคราะห์")
else:
    menu = st.sidebar.radio(
        "เลือกหัวข้อการวิเคราะห์", 
        ["👤 พรรณนา (Descriptive)", 
         "📊 สร้าง Epi Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Crude Analysis (OR/RR)", 
         "🧬 Adjusted Analysis (Logistic)",
         "📈 Dashboard & Report",
         "📝 ข้อมูลการลงทะเบียน (แก้ไข)"]
    )

# ==========================================
# 4. DATA SOURCE SELECTOR
# ==========================================
df = None
if st.session_state['registered']:
    st.sidebar.divider()
    source = st.sidebar.radio("แหล่งข้อมูล:", ["อัปโหลดไฟล์ (Excel/CSV)", "Google Sheets"])
    if source == "อัปโหลดไฟล์ (Excel/CSV)":
        uploaded_file = st.sidebar.file_uploader("📂 เลือกไฟล์ข้อมูล", type=['xlsx', 'csv'])
        if uploaded_file: df = load_data(uploaded_file)
    else:
        url = st.sidebar.text_input("🔗 ลิงก์ Google Sheets:")
        if url:
            conn = st.connection("gsheets", type=GSheetsConnection)
            df = conn.read(spreadsheet=url)

# ==========================================
# 5. MAIN CONTENT
# ==========================================

# --- หน้าลงทะเบียน ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
    with st.form("reg_v2"):
        u_agency = st.text_input("หน่วยงานต้นสังกัด", placeholder="เช่น รพ.สต.นาข่า")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรค", "ทำผลงานวิชาการ", "เรียนรู้/ซ้อมแผน"])
        if st.form_submit_button("เริ่มใช้งาน"):
            if u_agency:
                st.session_state['registered'] = True
                st.success("ลงทะเบียนสำเร็จ!")
                st.rerun()
            else: st.error("กรุณาระบุหน่วยงาน")

elif df is not None:
    total_n = len(df)

    # 1. ระบาดวิทยาเชิงพรรณนา (Descriptive)
    if menu == "👤 พรรณนา (Descriptive)":
        st.title("👤 ระบาดวิทยาเชิงพรรณนา")
        st.info(f"📋 จำนวนผู้ป่วยทั้งหมด (n) = {total_n} ราย")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1. เพศ (Sex)")
            sex_col = st.selectbox("ตัวแปรเพศ", df.columns, key="s1")
            sex_df = df[sex_col].value_counts().reset_index()
            sex_df.columns = ['เพศ', 'n']; sex_df['%'] = (sex_df['n']/total_n*100)
            st.table(sex_df.style.format({'%': '{:.2f}'}))

        with c2:
            st.subheader("2. อายุ (Age Groups)")
            age_col = st.selectbox("ตัวแปรอายุ", df.columns, key="s2")
            df['age_grp'] = pd.cut(df[age_col], bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
            age_df = df['age_grp'].value_counts().sort_index().reset_index()
            age_df.columns = ['กลุ่มอายุ', 'n']; age_df['%'] = (age_df['n']/total_n*100)
            st.table(age_df.style.format({'%': '{:.2f}'}))

        st.subheader("3. อาการและอาการแสดง (1=มีอาการ)")
        symp_cols = st.multiselect("เลือกตัวแปรอาการ", df.columns)
        if symp_cols:
            s_data = [{"อาการ": c, "n": int((df[c]==1).sum()), "%": ((df[c]==1).sum()/total_n*100)} for c in symp_cols]
            s_df = pd.DataFrame(s_data).sort_values("n", ascending=True)
            fig_s = px.bar(s_df, x="%", y="อาการ", orientation='h', title="ความถี่ของอาการ (ร้อยละ)", text_auto='.1f', color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_s, use_container_width=True)

    # 2. Epi Curve (Bin & Padding สมบูรณ์)
    elif menu == "📊 สร้าง Epi Curve (Time)":
        st.title("📊 Interactive Epidemic Curve")
        date_col = st.sidebar.selectbox("คอลัมน์วันเริ่มป่วย", df.columns)
        col_grp = st.sidebar.selectbox("แยกกลุ่มตามสี (Stacked):", ["<none>"] + list(df.columns))
        
        unit_map = {"Hour": "H", "Day": "D", "Week": "W", "Month": "M"}
        bin_size = st.sidebar.number_input("Bin Size", min_value=1, value=1)
        bin_unit = st.sidebar.selectbox("Unit", list(unit_map.keys()), index=1)
        freq = f"{bin_size}{unit_map[bin_unit]}"

        # ส่วน Padding หัว-ท้าย
        pad_before = st.sidebar.number_input(f"เพิ่มช่วงว่างก่อนหน้า ({bin_unit})", value=2)
        pad_after = st.sidebar.number_input(f"เพิ่มช่วงว่างข้างหลัง ({bin_unit})", value=2)

        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df_clean = df.dropna(subset=[date_col]).copy()

        if not df_clean.empty:
            # คำนวณช่วงเวลาแบบมี Padding
            start_dt = (df_clean[date_col].min() - pd.to_timedelta(pad_before * bin_size, unit=bin_unit[0].lower())).floor('D')
            end_dt = (df_clean[date_col].max() + pd.to_timedelta(pad_after * bin_size, unit=bin_unit[0].lower())).ceil('D')
            full_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)

            if col_grp == "<none>":
                counts = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size()
                chart_df = counts.reindex(full_range, fill_value=0).reset_index()
                chart_df.columns = [date_col, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', text_auto=True, color_discrete_sequence=['#89CFF0'])
            else:
                counts = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().unstack(fill_value=0)
                chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name='Cases')
                chart_df.columns = [date_col, col_grp, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp, color_discrete_sequence=px.colors.qualitative.Alphabet)

            fig.update_layout(bargap=0, xaxis_title="Onset Date", yaxis_title="Cases", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    # 3. Spot Map (Professional)
    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map Analytics")
        lat_c = next((c for c in df.columns if 'lat' in c.lower()), None)
        lon_c = next((c for c in df.columns if 'lon' in c.lower()), None)
        
        if lat_c and lon_c:
            df_map = df.dropna(subset=[lat_c, lon_c]).copy()
            m = folium.Map(location=[df_map[lat_c].mean(), df_map[lon_c].mean()], zoom_start=15)
            for idx, row in df_map.iterrows():
                folium.CircleMarker([row[lat_c], row[lon_c]], radius=7, color='red', fill=True).add_to(m)
            folium_static(m, width=1000)
        else: st.warning("ไม่พบคอลัมน์ Latitude/Longitude ในไฟล์ข้อมูล")

    # 4. Crude Analysis (OR/RR)
    elif menu == "🔬 Crude Analysis (OR/RR)":
        st.title("🔬 Bivariate Analysis (OR/RR)")
        out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns)
        exp_list = st.multiselect("เลือกปัจจัยเสี่ยง (Exposures)", [c for c in df.columns if c != out_v])
        
        if st.button("ประมวลผลสถิติ"):
            results = []
            for exp_v in exp_list:
                temp = df[[out_v, exp_v]].copy().dropna()
                temp[out_v], temp[exp_v] = smart_map_variable(temp[out_v]), smart_map_variable(temp[exp_v])
                a = len(temp[(temp[exp_v]==1) & (temp[out_v]==1)])
                b = len(temp[(temp[exp_v]==1) & (temp[out_v]==0)])
                c = len(temp[(temp[exp_v]==0) & (temp[out_v]==1)])
                d = len(temp[(temp[exp_v]==0) & (temp[out_v]==0)])
                or_val = (a*d)/(b*c) if (b*c) > 0 else 0
                mid_p = calculate_mid_p(a, b, c, d)
                results.append({"ปัจจัย": exp_v, "Odds Ratio": or_val, "Mid-P": mid_p})
            st.table(pd.DataFrame(results).style.format({"Odds Ratio": "{:.2f}", "Mid-P": "{:.4f}"}))

    # 5. Adjusted Analysis (Logistic Regression)
    elif menu == "🧬 Adjusted Analysis (Logistic)":
        st.title("🧬 Multiple Logistic Regression")
        out_v = st.selectbox("Outcome (Target)", df.columns, key="log_out")
        exp_v = st.selectbox("Main Exposure", [c for c in df.columns if c != out_v])
        adj_v = st.multiselect("Covariates (Adjusted)", [c for c in df.columns if c not in [out_v, exp_v]])
        
        if st.button("วิเคราะห์ Model"):
            try:
                df_m = df[[out_v, exp_v] + adj_v].copy().dropna()
                for c in df_m.columns: df_m[c] = smart_map_variable(df_m[c])
                formula = f"Q('{out_v}') ~ Q('{exp_v}') + " + " + ".join([f"Q('{a}')" for a in adj_v]) if adj_v else f"Q('{out_v}') ~ Q('{exp_v}')"
                model = smf.logit(formula, data=df_m).fit(disp=0)
                res_df = pd.DataFrame({"Adjusted OR": np.exp(model.params), "P-value": model.pvalues})
                st.dataframe(res_df.style.format("{:.3f}"))
            except Exception as e: st.error(f"Error: {e}")

else:
    st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้าย")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)
