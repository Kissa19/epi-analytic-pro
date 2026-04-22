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
    page_title="Epi-Analytic Pro | ODPC 8", 
    page_icon="🦠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ปรับโทนสีให้เป็นทางการ (Professional Grey Theme)
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #F8F9FB !important;
            border-right: 1px solid #EAEAEA;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 {
            color: #2C3E50 !important;
            font-weight: 500;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #f0f0f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'registered' not in st.session_state:
    st.session_state['registered'] = False

# ==========================================
# 3. HELPER FUNCTIONS (STATISTICS & DATA)
# ==========================================
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ ไม่สามารถอ่านไฟล์ได้: {e}")
        return None

def smart_map_variable(series):
    """แปลงค่า 1,2 ให้เป็น 1,0 อัตโนมัติสำหรับการวิเคราะห์สถิติ (1=มีปัจจัย/ป่วย, 0=ไม่มี/ไม่ป่วย)"""
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({1, 2, 1.0, 2.0, '1', '2'}):
        return pd.to_numeric(series, errors='coerce').map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    return series

def calculate_mid_p(a, b, c, d):
    """คำนวณ Mid-P Exact P-value แบบมาตรฐาน OpenEpi"""
    n = a + b + c + d
    if n == 0: return np.nan
    k = a + c # total sick
    m = a + b # total exposed
    p_obs = hypergeom.pmf(a, n, k, m)
    p_lower = hypergeom.cdf(a, n, k, m)
    p_upper = hypergeom.sf(a-1, n, k, m)
    mid_p = 2 * (min(p_lower, p_upper) - 0.5 * p_obs)
    return max(min(mid_p, 1.0), 0.0)

def find_col(df, possible_names):
    """ค้นหาชื่อคอลัมน์อัตโนมัติ"""
    return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
try:
    st.sidebar.image("odpc8_logo.png", use_container_width=True)
except:
    st.sidebar.markdown("<h2 style='text-align: center; color: #2C3E50;'>🏥 ODPC 8<br>Epi-Analytic Pro</h2>", unsafe_allow_html=True)

st.sidebar.markdown("---")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนก่อนเข้าใช้งาน\n\n🛡️ Data Privacy: ระบบไม่มีการเก็บบันทึกข้อมูลระบุตัวตน (PII)")
else:
    menu = st.sidebar.radio(
        "📌 เมนูการวิเคราะห์ (Analytics)", 
        [
            "👥 ประชากรและอัตราป่วย (Attack Rate)",
            "👤 บุคคล (Person)", 
            "📊 Epidemic Curve (Time)", 
            "🗺️ Spot Map (Place)",
            "🔬 Bivariate Analysis (OR/RR)", 
            "🧬 Multiple Logistic Regression (AOR)",
            "📝 ข้อมูลการลงทะเบียน (แก้ไข)"
        ]
    )

# ==========================================
# 5. DATA INGESTION
# ==========================================
df = None
if st.session_state['registered']:
    st.sidebar.divider()
    st.sidebar.subheader("💾 นำเข้าข้อมูล (Data Source)")
    data_source = st.sidebar.radio("เลือกแหล่งข้อมูล:", ["อัปโหลดไฟล์ (CSV/Excel)", "Google Sheets (API)"])

    if data_source == "อัปโหลดไฟล์ (CSV/Excel)":
        uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดชุดข้อมูล", type=['xlsx', 'csv'])
        if uploaded_file:
            df = load_data(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input("🔗 URL Google Sheets:")
        if sheet_url:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(spreadsheet=sheet_url)
                if st.sidebar.button("🔄 ซิงค์ข้อมูลล่าสุด"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error("❌ เชื่อมต่อล้มเหลว โปรดตรวจสอบว่าแชร์สิทธิ์ชีตเป็น 'Anyone with the link' หรือยัง")

# ==========================================
# 6. MAIN APPLICATION MODULES
# ==========================================

# --- MODULE: Registration ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
    st.info("💡 การลงทะเบียนนี้มีวัตถุประสงค์เพื่อเก็บสถิติการใช้งานนวัตกรรม สำหรับการวิจัย (ไม่เก็บข้อมูลส่วนบุคคล)")

    with st.form("registration_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_team = st.selectbox("ประเภททีมปฏิบัติการ", ["CDCU", "SRRT", "SAT", "JIT", "นักวิชาการ/นักระบาดวิทยา", "อื่นๆ"])
        with col2:
            u_agency = st.text_input("หน่วยงานต้นสังกัด (เช่น สสจ.อุดรธานี, รพ.เลย)")
            
        u_purpose = st.selectbox("วัตถุประสงค์การใช้งานหลัก", ["สนับสนุนการสอบสวนโรคภาคสนาม (Response)", "วิเคราะห์สถิติเพื่อจัดทำรายงาน (Reporting)", "ฝึกอบรม/ซ้อมแผน (Exercise)", "อื่นๆ"])
        
        submit_reg = st.form_submit_button("🚀 ยืนยันการเข้าสู่ระบบ")

        if submit_reg:
            if not u_agency:
                st.error("⚠️ กรุณาระบุหน่วยงานต้นสังกัด")
            else:
                from datetime import datetime
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                payload = {"timestamp": now, "team": u_team, "agency": u_agency, "purpose": u_purpose}
                try:
                    # Endpoint GAS ของ สคร.8
                    url = "https://script.google.com/macros/s/AKfycbxVGzrB9IjdvD90g2Zm8cKNwYE1PMrtaaun7YlBkGjWoL3UjVw74K49B_wg4cBfedeB/exec"
                    requests.post(url, json=payload, timeout=5)
                except:
                    pass # ยอมให้ผ่านถ้าอินเทอร์เน็ตมีปัญหา
                
                st.session_state['registered'] = True
                st.success("✅ เข้าสู่ระบบสำเร็จ ปลดล็อกเมนูการวิเคราะห์แล้ว")
                st.balloons()
                st.rerun()

# --- MODULES: Analytics (Requires Data) ---
elif st.session_state['registered']:
    if df is None:
        st.info("👈 กรุณานำเข้าข้อมูลดิบ (Data Source) ที่แถบเมนูด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์")
    else:
        total_n = len(df)

        # ------------------------------------------
        # 6.1 Attack Rate
        # ------------------------------------------
        if menu == "👥 ประชากรและอัตราป่วย (Attack Rate)":
            st.title("👥 ประชากรและอัตราป่วย (Attack Rate)")
            
            sex_c = find_col(df, ['sex', 'gender', 'เพศ'])
            age_c = find_col(df, ['age', 'อายุ'])

            with st.expander("⚙️ ระบุข้อมูลประชากรกลุ่มเสี่ยง (Denominator)", expanded=True):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**แยกตามเพศ**")
                    pop_male = st.number_input("ประชากรชาย (N)", min_value=1, value=100)
                    pop_female = st.number_input("ประชากรหญิง (N)", min_value=1, value=100)
                with col_p2:
                    st.markdown("**แยกตามกลุ่มอายุ**")
                    age_labels = ['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+']
                    pop_age = {lbl: st.number_input(f"กลุ่มอายุ {lbl}", min_value=0, value=0) for lbl in age_labels}

            if st.button("📈 คำนวณ Attack Rate", type="primary"):
                st.markdown("---")
                total_pop = pop_male + pop_female
                overall_ar = (len(df) / total_pop * 100) if total_pop > 0 else 0
                
                st.markdown(f"""
                <div class='metric-card' style='text-align: center; border-left: 5px solid #2980b9;'>
                    <h3 style='margin:0; color:#7f8c8d;'>Overall Attack Rate</h3>
                    <h1 style='margin:0; color:#2980b9;'>{overall_ar:.2f} %</h1>
                    <p style='margin:0; color:#95a5a6;'>จากผู้ป่วยทั้งหมด {len(df):,} ราย / ประชากร {total_pop:,} คน</p>
                </div>
                <br>
                """, unsafe_allow_html=True)

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown("#### Sex-Specific Attack Rate")
                    if sex_c:
                        df_sex = df.copy()
                        df_sex['sex_tmp'] = df_sex[sex_c].astype(str).str.strip().replace({'1':'ชาย','2':'หญิง','1.0':'ชาย','2.0':'หญิง'})
                        m_case = len(df_sex[df_sex['sex_tmp'] == 'ชาย'])
                        f_case = len(df_sex[df_sex['sex_tmp'] == 'หญิง'])
                        ar_sex_df = pd.DataFrame({
                            "เพศ": ["ชาย", "หญิง"], "ป่วย (n)": [m_case, f_case],
                            "ประชากร (N)": [pop_male, pop_female],
                            "AR (%)": [m_case/pop_male*100, f_case/pop_female*100]
                        })
                        st.dataframe(ar_sex_df.style.format({"AR (%)": "{:.2f}"}), use_container_width=True)
                    else: st.warning("ไม่พบคอลัมน์เพศ")
                    
                with res_col2:
                    st.markdown("#### Age-Specific Attack Rate")
                    if age_c:
                        df_age = df.copy()
                        df_age['age_tmp'] = pd.cut(pd.to_numeric(df_age[age_c], errors='coerce'), bins=[0,5,15,25,35,45,55,65,120], labels=age_labels, right=False)
                        age_counts = df_age['age_tmp'].value_counts().reindex(age_labels, fill_value=0)
                        ar_age_df = pd.DataFrame([{"กลุ่มอายุ": lbl, "ป่วย (n)": age_counts[lbl], "ประชากร (N)": pop_age[lbl], "AR (%)": (age_counts[lbl]/pop_age[lbl]*100) if pop_age[lbl]>0 else 0} for lbl in age_labels])
                        st.dataframe(ar_age_df.style.format({"AR (%)": "{:.2f}"}), use_container_width=True)

        # ------------------------------------------
        # 6.2 Person
        # ------------------------------------------
        elif menu == "👤 บุคคล (Person)":
            st.title("👤 สถิติเชิงพรรณนา (Descriptive: Person)")
            st.write(f"วิเคราะห์การกระจายตัวของผู้ป่วยจากฐานข้อมูล (N = {total_n})")
            
            c1, c2 = st.columns(2)
            with c1:
                sel_sex = st.selectbox("ตัวแปรเพศ", df.columns, index=0)
                res_sex = df[sel_sex].value_counts().reset_index()
                res_sex.columns = ['หมวดหมู่', 'จำนวน (n)']
                res_sex['ร้อยละ (%)'] = (res_sex['จำนวน (n)']/total_n*100)
                st.dataframe(res_sex.style.format({'ร้อยละ (%)': '{:.2f}'}), use_container_width=True)
                
            with c2:
                sel_age = st.selectbox("ตัวแปรอายุ", df.columns, index=min(1, len(df.columns)-1))
                df_clean = df.copy()
                df_clean['age_grp'] = pd.cut(pd.to_numeric(df_clean[sel_age], errors='coerce'), bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
                res_age = df_clean['age_grp'].value_counts().sort_index().reset_index()
                res_age.columns = ['กลุ่มอายุ', 'จำนวน (n)']
                res_age['ร้อยละ (%)'] = (res_age['จำนวน (n)']/total_n*100)
                st.dataframe(res_age.style.format({'ร้อยละ (%)': '{:.2f}'}), use_container_width=True)

        # ------------------------------------------
        # 6.3 Epidemic Curve
        # ------------------------------------------
        elif menu == "📊 Epidemic Curve (Time)":
            st.title("📊 Epidemic Curve (Time)")
            date_col = st.selectbox("เลือกคอลัมน์วันเวลาเริ่มป่วย (Onset)", df.columns)
            
            # Smart Parsing Date
            df_time = df.copy()
            df_time[date_col] = pd.to_datetime(df_time[date_col], dayfirst=True, errors='coerce')
            df_time = df_time.dropna(subset=[date_col])
            
            if df_time.empty:
                st.error("❌ ไม่สามารถแปลงรูปแบบวันที่ได้ กรุณาตรวจสอบข้อมูลในไฟล์")
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    freq_map = {"รายวัน (Day)": "D", "รายชั่วโมง (Hour)": "H", "รายสัปดาห์ (Week)": "W-MON"}
                    freq_choice = st.radio("ความละเอียดของแกนเวลา (Bin):", list(freq_map.keys()))
                    grp_col = st.selectbox("แยกสีตามตัวแปร (Optional):", ["<ไม่มี>"] + list(df.columns))
                
                with c2:
                    freq = freq_map[freq_choice]
                    if grp_col == "<ไม่มี>":
                        counts = df_time.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name='Cases')
                        fig = px.bar(counts, x=date_col, y='Cases', text_auto=True, color_discrete_sequence=['#2980b9'])
                    else:
                        counts = df_time.groupby([pd.Grouper(key=date_col, freq=freq), grp_col]).size().reset_index(name='Cases')
                        fig = px.bar(counts, x=date_col, y='Cases', color=grp_col, color_discrete_sequence=px.colors.qualitative.Set2)
                    
                    fig.update_layout(xaxis_title="เวลาที่เริ่มป่วย (Onset Date)", yaxis_title="จำนวนผู้ป่วย (Cases)", bargap=0.05, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # 6.4 Spot Map
        # ------------------------------------------
        elif menu == "🗺️ Spot Map (Place)":
            st.title("🗺️ Spot Map Analytics")
            lat_c = find_col(df, ['lat', 'latitude', 'ละติจูด'])
            lon_c = find_col(df, ['lon', 'longitude', 'ลองจิจูด'])
            
            if lat_c and lon_c:
                df_map = df.copy()
                df_map[lat_c] = pd.to_numeric(df_map[lat_c], errors='coerce')
                df_map[lon_c] = pd.to_numeric(df_map[lon_c], errors='coerce')
                df_map = df_map.dropna(subset=[lat_c, lon_c])
                
                if df_map.empty:
                    st.error("ไม่พบพิกัดที่สามารถพล็อตได้")
                else:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown("#### ตั้งค่าแผนที่")
                        radius = st.selectbox("รัศมีวงรอบ (Buffer Zone)", [0, 50, 100, 200, 500])
                        color_var = st.selectbox("แยกสีตามตัวแปร", ["<สีแดงมาตรฐาน>"] + list(df.columns))
                    
                    with c2:
                        m = folium.Map(location=[df_map[lat_c].mean(), df_map[lon_c].mean()], zoom_start=15, tiles='OpenStreetMap')
                        for idx, r in df_map.iterrows():
                            # Logic แยกสี
                            c_val = "red"
                            if color_var != "<สีแดงมาตรฐาน>":
                                # ใช้สีสุ่มเบาๆ หรือกำหนดตามค่า
                                c_val = "blue" if str(r[color_var]) in ['1', 'ชาย', 'Yes'] else "orange"

                            popup_html = f"<b>เคสที่:</b> {idx+1}<br><b>พิกัด:</b> {r[lat_c]}, {r[lon_c]}"
                            folium.CircleMarker([r[lat_c], r[lon_col]], radius=6, color=c_val, fill=True, fill_opacity=0.8, popup=popup_html).add_to(m)
                            if radius > 0:
                                folium.Circle([r[lat_c], r[lon_c]], radius=radius, color='gray', weight=1, fill=True, fill_opacity=0.1).add_to(m)
                        
                        folium_static(m, width=900, height=500)
            else:
                st.warning("⚠️ ไม่พบคอลัมน์พิกัด (Latitude/Longitude) กรุณาตรวจสอบ Header ในไฟล์ Excel")

        # ------------------------------------------
        # 6.5 Bivariate Analysis
        # ------------------------------------------
        elif menu == "🔬 Bivariate Analysis (OR/RR)":
            st.title("🔬 Bivariate Analysis (2x2 Table)")
            tab1, tab2 = st.tabs(["📁 วิเคราะห์แบบกลุ่มจากไฟล์ (Batch)", "🔢 คำนวณตาราง 2x2 Manual"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    out_v = st.selectbox("เลือกตัวแปรตาม (Outcome)", df.columns, help="ตัวแปรโรค เช่น ป่วย/ไม่ป่วย (1/0)")
                with col2:
                    exp_list = st.multiselect("เลือกปัจจัยความเสี่ยง (Exposures)", [c for c in df.columns if c != out_v])
                
                if st.button("🚀 ประมวลผลสถิติ", type="primary"):
                    if not exp_list:
                        st.error("กรุณาเลือกปัจจัยเสี่ยงอย่างน้อย 1 ตัว")
                    else:
                        results = []
                        for exp_v in exp_list:
                            temp = df[[out_v, exp_v]].copy().dropna()
                            temp[out_v] = smart_map_variable(temp[out_v])
                            temp[exp_v] = smart_map_variable(temp[exp_v])
                            
                            a = len(temp[(temp[exp_v]==1) & (temp[out_v]==1)])
                            b = len(temp[(temp[exp_v]==1) & (temp[out_v]==0)])
                            c = len(temp[(temp[exp_v]==0) & (temp[out_v]==1)])
                            d = len(temp[(temp[exp_v]==0) & (temp[out_v]==0)])
                            
                            or_val = (a*d)/(b*c) if (b*c)>0 else np.nan
                            
                            # 95% CI
                            try:
                                se_ln = math.sqrt(1/a + 1/b + 1/c + 1/d) if a*b*c*d > 0 else np.nan
                                lower = math.exp(math.log(or_val) - 1.96 * se_ln) if not np.isnan(or_val) and or_val>0 else np.nan
                                upper = math.exp(math.log(or_val) + 1.96 * se_ln) if not np.isnan(or_val) and or_val>0 else np.nan
                            except: lower, upper = np.nan, np.nan

                            mid_p = calculate_mid_p(a,b,c,d)
                            results.append({"ปัจจัย": exp_v, "ป่วย(+)": a, "ไม่ป่วย(+)": b, "ป่วย(-)": c, "ไม่ป่วย(-)": d, "Odds Ratio": or_val, "95% CI": f"{lower:.2f} - {upper:.2f}" if not np.isnan(lower) else "-", "Mid-P": mid_p})
                        
                        res_df = pd.DataFrame(results)
                        st.dataframe(res_df.style.format({"Odds Ratio": "{:.2f}", "Mid-P": "{:.5f}"}).apply(lambda x: ['background-color: #d4edda' if x['Mid-P'] < 0.05 else '' for _ in x], axis=1), use_container_width=True)
                        st.caption("✨ ไฮไลต์สีเขียว หมายถึง มีนัยสำคัญทางสถิติ (p-value < 0.05)")

            with tab2:
                c1, c2, c3 = st.columns([1,1,1])
                with c2: ma = st.number_input("ป่วย + สัมผัส (a)", 0, value=0)
                with c3: mb = st.number_input("ไม่ป่วย + สัมผัส (b)", 0, value=0)
                with c2: mc = st.number_input("ป่วย + ไม่สัมผัส (c)", 0, value=0)
                with c3: md = st.number_input("ไม่ป่วย + ไม่สัมผัส (d)", 0, value=0)
                
                if st.button("คำนวณ Manual"):
                    or_m = (ma*md)/(mb*mc) if (mb*mc)>0 else 0
                    p_m = calculate_mid_p(ma, mb, mc, md)
                    st.success(f"**Odds Ratio:** {or_m:.2f} | **Mid-P Exact:** {p_m:.5f}")

        # ------------------------------------------
        # 6.6 Multiple Logistic Regression
        # ------------------------------------------
        elif menu == "🧬 Multiple Logistic Regression (AOR)":
            st.title("🧬 Multiple Logistic Regression")
            st.write("วิเคราะห์หาค่า Adjusted Odds Ratio (AOR) โดยควบคุมตัวแปรกวน")
            
            c1, c2 = st.columns(2)
            with c1:
                out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns)
                exp_v = st.selectbox("ปัจจัยหลักที่สนใจ (Main Exposure)", [c for c in df.columns if c != out_v])
            with c2:
                adj_v = st.multiselect("ตัวแปรกวน (Confounders)", [c for c in df.columns if c not in [out_v, exp_v]])
            
            if st.button("🚀 ประมวลผลโมเดล (Run Model)", type="primary"):
                try:
                    # Clean Data
                    cols_model = [out_v, exp_v] + adj_v
                    df_model = df[cols_model].copy().dropna()
                    for col in df_model.columns:
                        df_model[col] = smart_map_variable(df_model[col])
                    
                    # Formula formulation
                    formula = f"Q('{out_v}') ~ Q('{exp_v}')"
                    if adj_v: formula += " + " + " + ".join([f"Q('{a}')" for a in adj_v])
                    
                    # Fit Model
                    model = smf.logit(formula, data=df_model).fit(disp=0)
                    conf = model.conf_int()
                    
                    # Create Output Dataframe
                    res_df = pd.DataFrame({
                        "Variable": model.params.index,
                        "Adjusted OR": np.exp(model.params.values),
                        "95% CI Lower": np.exp(conf[0].values),
                        "95% CI Upper": np.exp(conf[1].values),
                        "P-value": model.pvalues.values
                    })
                    
                    # Clean Labels
                    res_df = res_df[res_df['Variable'] != 'Intercept']
                    res_df['Variable'] = res_df['Variable'].str.extract(r"Q\('(.*)'\)")[0].fillna(res_df['Variable'])
                    
                    st.subheader("📋 ผลการวิเคราะห์โมเดล (Model Summary)")
                    st.dataframe(res_df.style.format({
                        "Adjusted OR": "{:.2f}",
                        "95% CI Lower": "{:.2f}",
                        "95% CI Upper": "{:.2f}",
                        "P-value": "{:.4f}"
                    }), use_container_width=True)
                    st.success(f"✅ ประมวลผลสำเร็จจากชุดข้อมูล {len(df_model)} รายการ (ที่ข้อมูลครบถ้วน)")
                except Exception as e:
                    st.error(f"❌ ไม่สามารถสร้างโมเดลได้: อาจมีค่าว่าง หรือผู้ป่วย/ไม่ป่วยในกลุ่มมีจำนวน 0 (Perfect separation) รายละเอียด: {e}")

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #95a5a6; font-size: 13px;'>Epi-Analytic Pro v.2.0 | พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)
