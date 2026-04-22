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
import math

# ==========================================
# 1. CONFIGURATION & STYLING
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
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] .st-at {
            color: #31333F !important;
            font-weight: 500;
        }
        .stButton > button { border-radius: 8px; }
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
        if file.name.endswith('.csv'): return pd.read_csv(file)
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
except:
    st.sidebar.title("🏥 ODPC8 Udon Thani")

st.sidebar.markdown("---")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนเพื่อปลดล็อกเมนูวิเคราะห์")
else:
    menu = st.sidebar.radio(
        "เลือกหัวข้อการวิเคราะห์", 
        ["👥 ประชากรและอัตราป่วย (Attack Rate)",
         "👤 พรรณนา (Descriptive)", 
         "📊 สร้าง Epi Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Bivariate Analysis (OR/RR)", 
         "🧬 Multiple Logistic Regression (AOR)",
         "📝 ข้อมูลการลงทะเบียน (แก้ไข)"]
    )

# ==========================================
# 5. DATA SOURCE MANAGEMENT
# ==========================================
df = None
if st.session_state['registered']:
    st.sidebar.divider()
    source_choice = st.sidebar.radio("แหล่งข้อมูล:", ["อัปโหลดไฟล์ (Excel/CSV)", "Google Sheets"])
    
    if source_choice == "อัปโหลดไฟล์ (Excel/CSV)":
        uploaded_file = st.sidebar.file_uploader("📂 เลือกไฟล์ข้อมูล", type=['xlsx', 'csv'])
        if uploaded_file:
            df = load_data(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input("🔗 ลิงก์ Google Sheets:")
        if sheet_url:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(spreadsheet=sheet_url)
                if st.sidebar.button("🔄 อัปเดตข้อมูล"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"เชื่อมต่อล้มเหลว: {e}")

# ==========================================
# 6. MAIN CONTENT
# ==========================================

# --- หน้าลงทะเบียน ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
    with st.form("registration"):
        u_agency = st.text_input("หน่วยงานต้นสังกัด (เช่น สสจ.อุดรธานี)")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรคภาคสนาม", "วิเคราะห์สถิติวิชาการ", "ซ้อมแผนฯ"])
        if st.form_submit_button("เริ่มใช้งาน"):
            if u_agency:
                from datetime import datetime
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                payload = {"timestamp": now, "agency": u_agency, "purpose": u_purpose}
                try:
                    url = "https://script.google.com/macros/s/AKfycbxVGzrB9IjdvD90g2Zm8cKNwYE1PMrtaaun7YlBkGjWoL3UjVw74K49B_wg4cBfedeB/exec"
                    requests.post(url, json=payload, timeout=5)
                except: pass
                st.session_state['registered'] = True
                st.success("ลงทะเบียนสำเร็จ!")
                st.rerun()
            else: st.error("กรุณาระบุหน่วยงาน")

elif df is not None:
    total_n = len(df)

    # 1. Attack Rate
    if menu == "👥 ประชากรและอัตราป่วย (Attack Rate)":
        st.title("👥 ประชากรและอัตราป่วย (Attack Rate)")
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
            ar = (total_n / (pop_male + pop_female) * 100)
            st.metric("Overall Attack Rate", f"{ar:.2f} %")
            # ... (ตารางย่อยตามที่เคยทำ)

    # 2. Descriptive Analysis
    elif menu == "👤 พรรณนา (Descriptive)":
        st.title("👤 ระบาดวิทยาเชิงพรรณนา")
        st.info(f"📋 จำนวนผู้ป่วยทั้งหมด (n) = {total_n} ราย")
        
        c1, c2 = st.columns(2)
        with c1:
            sex_col = st.selectbox("ตัวแปรเพศ", df.columns)
            res_sex = df[sex_col].value_counts().reset_index()
            res_sex.columns = ['เพศ', 'n']; res_sex['%'] = (res_sex['n']/total_n*100)
            st.table(res_sex.style.format({'%': '{:.2f}'}))
        with c2:
            age_col = st.selectbox("ตัวแปรอายุ", df.columns)
            df['age_grp'] = pd.cut(pd.to_numeric(df[age_col], errors='coerce'), bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
            res_age = df['age_grp'].value_counts().sort_index().reset_index()
            res_age.columns = ['อายุ', 'n']; res_age['%'] = (res_age['n']/total_n*100)
            st.table(res_age.style.format({'%': '{:.2f}'}))

        st.subheader("อาการแสดง (1=มี)")
        symp_cols = st.multiselect("เลือกตัวแปรอาการ", df.columns)
        if symp_cols:
            s_df = pd.DataFrame([{"อาการ": c, "%": (df[c]==1).sum()/total_n*100} for c in symp_cols]).sort_values("%", ascending=True)
            st.plotly_chart(px.bar(s_df, x="%", y="อาการ", orientation='h', text_auto='.1f'), use_container_width=True)

    # 3. Epidemic Curve (Advanced Fix)
    elif menu == "📊 สร้าง Epi Curve (Time)":
        st.title("📊 Interactive Epidemic Curve (Advanced)")
        date_col = st.sidebar.selectbox("คอลัมน์วันเริ่มป่วย", df.columns)
        col_grp = st.sidebar.selectbox("ตัวแปรแยกกลุ่มสี:", ["<none>"] + list(df.columns))
        
        unit_map = {"Hour": "h", "Day": "d", "Week": "W", "Month": "ME", "30 Min": "30min"}
        bin_unit = st.sidebar.selectbox("หน่วยเวลา", list(unit_map.keys()), index=0)
        bin_size = st.sidebar.number_input("ขนาด Bin", min_value=1, value=1)
        freq = f"{bin_size}{unit_map[bin_unit]}"

        pad_before = st.sidebar.number_input(f"เพิ่มช่วงก่อนหน้า ({bin_unit})", value=1)
        pad_after = st.sidebar.number_input(f"เพิ่มช่วงข้างหลัง ({bin_unit})", value=1)

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
                fig = px.bar(chart_df, x=date_col, y='Cases', text_auto=True, color_discrete_sequence=['#3498db'])
            else:
                counts = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().unstack(fill_value=0)
                chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name='Cases')
                chart_df.columns = [date_col, col_grp, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp)

            fig.update_layout(bargap=0.01, xaxis=dict(type='date', tickformat='%d/%m %H:%M'))
            st.plotly_chart(fig, use_container_width=True)

    # 4. Spot Map
    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map")
        lat_c = find_col(df, ['lat', 'latitude', 'ละติจูด'])
        lon_c = find_col(df, ['lon', 'longitude', 'ลองจิจูด'])
        if lat_c and lon_c:
            df_m = df.dropna(subset=[lat_c, lon_c])
            m = folium.Map(location=[df_m[lat_c].mean(), df_m[lon_c].mean()], zoom_start=15)
            for _, r in df_m.iterrows():
                folium.CircleMarker([r[lat_c], r[lon_c]], radius=7, color='red', fill=True).add_to(m)
            folium_static(m, width=1000)

    # 5. Bivariate & Logistic
    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        st.title("🔬 Bivariate Analysis")
        out_v = st.selectbox("Outcome", df.columns)
        exp_list = st.multiselect("Exposures", [c for c in df.columns if c != out_v])
        if st.button("🚀 ประมวลผล"):
            results = []
            for e in exp_list:
                temp = df[[out_v, e]].copy().dropna()
                temp[out_v], temp[e] = smart_map_variable(temp[out_v]), smart_map_variable(temp[e])
                a = len(temp[(temp[e]==1) & (temp[out_v]==1)])
                b = len(temp[(temp[e]==1) & (temp[out_v]==0)])
                c = len(temp[(temp[e]==0) & (temp[out_v]==1)])
                d = len(temp[(temp[e]==0) & (temp[out_v]==0)])
                or_val = (a*d)/(b*c) if (b*c)>0 else 0
                results.append({"ปัจจัย": e, "OR": or_val, "Mid-P": calculate_mid_p(a,b,c,d)})
            st.table(pd.DataFrame(results).style.format({"OR": "{:.2f}", "Mid-P": "{:.4f}"}))

    elif menu == "🧬 Multiple Logistic Regression (AOR)":
        st.title("🧬 Multiple Logistic Regression")
        out_v = st.selectbox("Outcome", df.columns, key="mlr_out")
        exp_v = st.selectbox("ปัจจัยหลัก", [c for c in df.columns if c != out_v])
        adj_v = st.multiselect("ตัวแปรกวน", [c for c in df.columns if c not in [out_v, exp_v]])
        if st.button("🚀 คำนวณ AOR"):
            try:
                df_m = df[[out_v, exp_v] + adj_v].copy().dropna()
                for c in df_m.columns: df_m[c] = smart_map_variable(df_m[c])
                formula = f"Q('{out_v}') ~ Q('{exp_v}')" + (" + " + " + ".join([f"Q('{a}')" for a in adj_v]) if adj_v else "")
                model = smf.logit(formula, data=df_m).fit(disp=0)
                res = pd.DataFrame({"AOR": np.exp(model.params), "P-value": model.pvalues})
                st.table(res[res.index != 'Intercept'].style.format("{:.3f}"))
            except Exception as e: st.error(f"Error: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Epi-Analytic Pro ODPC8 | พัฒนาโดย กลุ่มระบาดวิทยา</div>", unsafe_allow_html=True)
