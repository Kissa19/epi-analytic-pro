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
import re

# ==========================================
# 1. CONFIGURATION & STYLING (DDC PINK-WHITE)
# ==========================================
st.set_page_config(
    page_title="Epi-Analytic Pro ODPC8", 
    page_icon="🦠", 
    layout="wide"
)

# บังคับฟอนต์ Kanit และธีมสีกรมควบคุมโรค (ลดขนาด h1, h2 ให้สมดุล)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        /* บังคับฟอนต์ Kanit ให้ครอบคลุมทุก Element บนหน้าเว็บ */
        html, body, [class*="css"], [class*="st-"], div, span, applet, object, iframe,
        h1, h2, h3, h4, h5, h6, p, blockquote, pre, a, abbr, acronym, address, big, cite, code,
        del, dfn, em, img, ins, kbd, q, s, samp, small, strike, strong, sub, sup, tt, var,
        b, u, i, center, dl, dt, dd, ol, ul, li, fieldset, form, label, legend,
        table, caption, tbody, tfoot, thead, tr, th, td, article, aside, canvas, details, embed, 
        figure, figcaption, footer, header, hgroup, menu, nav, output, ruby, section, summary,
        time, mark, audio, video, button, input, select, textarea {
            font-family: 'Kanit', sans-serif !important;
        }

        /* ธีมสี Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFF0F5 !important; 
            border-right: 1px solid #F8BBD0;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #880E4F !important;
        }
        
        /* ปุ่มกด */
        .stButton > button {
            background-color: #E91E63 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: none !important;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #C2185B !important;
        }

        /* ปรับสมดุลขนาดตัวอักษร */
        h1 { font-size: 2.0rem !important; color: #D81B60 !important; font-weight: 600 !important; padding-bottom: 0.5rem; }
        h2 { font-size: 1.6rem !important; color: #D81B60 !important; font-weight: 500 !important; }
        h3 { font-size: 1.2rem !important; color: #880E4F !important; font-weight: 500 !important; }
        p, span, label, div { font-size: 0.95rem !important; }
        
        /* ตัวเลข Metric */
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #E91E63 !important; font-weight: 600 !important; }
        [data-testid="stMetricLabel"] { font-size: 1rem !important; font-weight: 500 !important; color: #666 !important; }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { font-size: 0.9rem !important; }

        /* สไตล์กล่องดาวน์โหลดไฟล์ตัวอย่าง */
        .template-box {
            background-color: #ffffff;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #F8BBD0;
            margin-bottom: 10px;
        }
        .template-link {
            color: #E91E63 !important;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
        }
        .template-link:hover {
            text-decoration: underline;
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
    # เพิ่ม key="main_menu_radio" เพื่อแก้ปัญหา DuplicateElementId
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
    # เพิ่ม key="data_source_radio" 
    source_choice = st.sidebar.radio(
        "เลือกแหล่งข้อมูล:", 
        ["อัปโหลดไฟล์ (Excel/CSV)", "Google Sheets"],
        key="data_source_radio"
    )
    
    if source_choice == "อัปโหลดไฟล์ (Excel/CSV)":
        uploaded_file = st.sidebar.file_uploader("📂 เลือกไฟล์ข้อมูล", type=['xlsx', 'csv'])
        if uploaded_file:
            df = load_data(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input("🔗 ลิงก์ Google Sheets:")
        if sheet_url:
            try:
                # สกัด ID และใช้วิธี Export เป็น CSV เพื่อป้องกัน HTTP 404 Error
                if "docs.google.com/spreadsheets" in sheet_url:
                    match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
                    if match:
                        sheet_id = match.group(1)
                        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                        df = pd.read_csv(csv_url)
                    else:
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        df = conn.read(spreadsheet=sheet_url)
                else:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    df = conn.read(spreadsheet=sheet_url)
                    
                if st.sidebar.button("🔄 อัปเดตข้อมูล"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"เชื่อมต่อล้มเหลว: {e}")
                st.info("💡 คำแนะนำ: โปรดตรวจสอบว่าลิงก์ Google Sheets เปิดสิทธิ์การแชร์เป็น 'ทุกคนที่มีลิงก์' (Anyone with the link) แล้วหรือไม่")

    # --- ลิงก์ไฟล์ตัวอย่าง ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 ไฟล์ตัวอย่าง (Templates)")
    st.sidebar.markdown(f"""
    <div class="template-box">
        <p style="margin-bottom:5px; font-size:0.85rem; color:#666;">ดาวน์โหลดไฟล์สำหรับทดลองระบบ:</p>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/13P9k7ucYHjbNQ88EucKXnR7JvPwGLEHF/edit?usp=drive_link" target="_blank">📄 1. พรรณนา/Daily Curve/Spot Map</a><br><br>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1kZSskpErufY_9qTl-_1TZaVymGMnNikm/edit?usp=drive_link" target="_blank">🕒 2. Hourly Epidemic Curve</a><br><br>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1TPJDOoIWCiZBtsnXDlhcHcN5IM27TBOK/edit?usp=drive_link" target="_blank">🔬 3. Case Control Analysis</a><br><br>
        <a class="template-link" href="https://docs.google.com/spreadsheets/d/1HR57-mVqo9TceAgF1tpzWvLQi662akzw/edit?usp=drive_link" target="_blank">📊 4. Cohort Study Analysis</a>
    </div>
    """, unsafe_allow_html=True)

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

    # ------------------------------------------
    # 6.1 Attack Rate
    # ------------------------------------------
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
            total_pop = pop_male + pop_female
            ar = (total_n / total_pop * 100) if total_pop > 0 else 0
            st.metric("Overall Attack Rate", f"{ar:.2f} %")
            
            c_res1, c_res2 = st.columns(2)
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
            with c_res2:
                st.markdown("**Age-Specific Attack Rate**")
                if age_c:
                    df['age_tmp'] = pd.cut(df[age_c], bins=[0,5,15,25,35,45,55,65,120], labels=age_labels, right=False)
                    a_cases = df['age_tmp'].value_counts().reindex(age_labels, fill_value=0)
                    ar_age = [{"อายุ": l, "ป่วย": a_cases[l], "ประชากร": pop_age[l], "AR (%)": (a_cases[l]/pop_age[l]*100) if pop_age[l]>0 else 0} for l in age_labels]
                    st.table(pd.DataFrame(ar_age).style.format({"AR (%)": "{:.2f}"}))

    # ------------------------------------------
    # 6.2 Descriptive Analysis
    # ------------------------------------------
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

        st.subheader("อาการแสดง (1=มีอาการ)")
        symp_cols = st.multiselect("เลือกตัวแปรอาการ", df.columns)
        if symp_cols:
            s_df = pd.DataFrame([{"อาการ": c, "%": (df[c]==1).sum()/total_n*100} for c in symp_cols]).sort_values("%", ascending=True)
            st.plotly_chart(px.bar(s_df, x="%", y="อาการ", orientation='h', text_auto='.1f', color_discrete_sequence=['#E91E63']), use_container_width=True)

    # ------------------------------------------
    # 6.3 Epidemic Curve (Advanced Range)
    # ------------------------------------------
    elif menu == "📊 สร้าง Epi Curve (Time)":
        st.title("📊 Interactive Epidemic Curve")
        date_col = st.sidebar.selectbox("คอลัมน์วันเริ่มป่วย", df.columns)
        col_grp = st.sidebar.selectbox("ตัวแปรแยกกลุ่มสี:", ["<none>"] + list(df.columns))
        
        unit_map = {"Hour": "h", "Day": "d", "Week": "W", "Month": "ME", "30 Min": "30min"}
        bin_unit = st.sidebar.selectbox("หน่วยเวลา", list(unit_map.keys()), index=0)
        bin_size = st.sidebar.number_input("ขนาด Bin", min_value=1, value=1)
        freq = f"{bin_size}{unit_map[bin_unit]}"

        pad_before = st.sidebar.number_input(f"เพิ่มช่วงว่างก่อนหน้า ({bin_unit})", value=1)
        pad_after = st.sidebar.number_input(f"เพิ่มช่วงว่างข้างหลัง ({bin_unit})", value=1)

        # แปลงวันที่
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
                fig = px.bar(chart_df, x=date_col, y='Cases', text_auto=True, color_discrete_sequence=['#E91E63'])
            else:
                counts = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().unstack(fill_value=0)
                chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name='Cases')
                chart_df.columns = [date_col, col_grp, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp, color_discrete_sequence=px.colors.sequential.RdPu[::-1])

            fig.update_layout(
                bargap=0.01, 
                xaxis=dict(type='date', tickformat='%d/%m %H:%M'),
                xaxis_title="Onset Date/Time",
                yaxis_title="Number of Cases",
                hovermode="x unified"
            )
            fig.update_traces(marker_line_width=0.5, marker_line_color='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ ไม่สามารถวิเคราะห์ได้ เนื่องจากรูปแบบวันที่ในไฟล์ไม่ถูกต้อง")

    # ------------------------------------------
    # 6.4 Spot Map (Satellite & Buffer Zone)
    # ------------------------------------------
    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map - GIS Analytics")
        lat_c = next((c for c in df.columns if any(p in c.lower() for p in ['lat', 'latitude', 'ละติจูด'])), None)
        lon_c = next((c for c in df.columns if any(p in c.lower() for p in ['lon', 'longitude', 'ลองจิจูด'])), None)
        
        if lat_c and lon_c:
            df_m = df.dropna(subset=[lat_c, lon_c]).copy()

            st.sidebar.markdown("---")
            st.sidebar.subheader("⚙️ ตั้งค่าแผนที่")
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
                # รัศมีควบคุมโรคหน่วยเป็นเมตร
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

                # จุดพิกัดผู้ป่วย
                folium.CircleMarker(
                    location=[r[lat_c], r[lon_c]], 
                    radius=6, 
                    color='#E91E63',
                    fill=True, 
                    fill_opacity=1.0,
                    popup=f"เคสที่ {idx+1}"
                ).add_to(m)

            folium_static(m, width=1000, height=650)
        else: 
            st.warning("⚠️ ไม่พบคอลัมน์พิกัด (Lat/Lon) ในไฟล์ กรุณาตรวจสอบชื่อคอลัมน์")

    # ------------------------------------------
    # 6.5 Bivariate Analysis
    # ------------------------------------------
    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        st.title("🔬 Bivariate Analysis & 2x2 Table")

        # สร้าง Tab เพื่อแยกการวิเคราะห์แบบไฟล์ และแบบกรอกเอง
        tab1, tab2 = st.tabs(["📁 วิเคราะห์จากไฟล์ข้อมูล", "🔢 กรอกข้อมูลเอง (Manual 2x2)"])

        with tab1:
            st.subheader("📁 วิเคราะห์ปัจจัยเสี่ยงจากไฟล์ที่อัปโหลด")
            if df is not None:
                out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns, key="file_out")
                design = st.radio("ประเภทการศึกษา", ["Case-control Study (OR)", "Cohort Study (RR)"], key="file_design")
                exp_list = st.multiselect("เลือกปัจจัยเสี่ยง", [c for c in df.columns if c != out_v], key="file_exp")

                if st.button("🚀 ประมวลผลจากไฟล์"):
                    import math
                    from scipy.stats import hypergeom
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
                                # 1. คำนวณ Point Estimate และ 95% CI (Taylor Series)
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

                                # 2. คำนวณ Mid-P Exact P-value (2-tail)
                                def calc_mid_p(a, b, c, d):
                                    n = a + b + c + d
                                    k = a + c # total sick
                                    m = a + b # total exposed
                                    if n == 0 or k == 0 or m == 0: return 1.0
                                    p_obs = hypergeom.pmf(a, n, k, m)
                                    p_lower = hypergeom.cdf(a, n, k, m)
                                    p_upper = hypergeom.sf(a-1, n, k, m)
                                    return 2 * (min(p_lower, p_upper) - 0.5 * p_obs)

                                mid_p_val = calc_mid_p(a, b, c, d)

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
                    else:
                        st.warning("⚠️ ไม่พบข้อมูลที่เพียงพอในการวิเคราะห์")

        with tab2:
            st.subheader("🔢 Manual 2x2 Table Calculator")
            st.info("ใช้สำหรับคำนวณกรณีมีเพียงตัวเลขสรุป (Aggregated Data) โดยไม่ต้องอัปโหลดไฟล์")

            # 1. เลือกรูปแบบการศึกษา
            manual_design = st.radio(
                "รูปแบบการศึกษา (Study Design):",
                ["Cohort Study (Relative Risk)", "Case-Control Study (Odds Ratio)"],
                horizontal=True, key="man_design"
            )

            # 2. ส่วนการกรอกข้อมูล 2x2 Table
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

            # 3. ส่วนการคำนวณสถิติ
            if st.button("📈 คำนวณผล 2x2 Table"):
                if (ma + mb + mc + md) > 0:
                    import math
                    from scipy.stats import chi2_contingency, hypergeom

                    try:
                        # --- คำนวณค่า Point Estimate ---
                        if "Case-Control" in manual_design:
                            res_label = "Odds Ratio (OR)"
                            val = (ma * md) / (mb * mc) if (mb * mc) > 0 else 0
                        else:
                            res_label = "Relative Risk (RR)"
                            val = (ma / (ma + mb)) / (mc / (mc + md)) if (ma + mb) > 0 and (mc + md) > 0 else 0

                        # --- คำนวณ 95% CI แบบ Taylor Series (มาตรฐาน OpenEpi) ---
                        if "Case-Control" in manual_design:
                            # Taylor Series for OR
                            se_ln = math.sqrt(1/ma + 1/mb + 1/mc + 1/md)
                        else:
                            # Taylor Series for RR
                            se_ln = math.sqrt((1/ma - 1/(ma+mb)) + (1/mc - 1/(mc+md)))

                        lower = math.exp(math.log(val) - 1.96 * se_ln) if val > 0 else 0
                        upper = math.exp(math.log(val) + 1.96 * se_ln) if val > 0 else 0

                        # --- คำนวณ Chi-Square (Yates และ Uncorrected) ---
                        obs = np.array([[ma, mb], [mc, md]])
                        chi2_uncorrected, p_uncor, _, _ = chi2_contingency(obs, correction=False)
                        chi2_yates, p_yates, _, _ = chi2_contingency(obs, correction=True)

                        # --- คำนวณ Mid-P Exact P-value (2-tail) ---
                        def get_mid_p(a, b, c, d):
                            n = a + b + c + d
                            k = a + c # total sick
                            m = a + b # total exposed
                            p_obs = hypergeom.pmf(a, n, k, m)
                            # Mid-P = P(extreme) - 0.5 * P(observed)
                            p_lower = hypergeom.cdf(a, n, k, m)
                            p_upper = hypergeom.sf(a-1, n, k, m)
                            return 2 * (min(p_lower, p_upper) - 0.5 * p_obs)

                        mid_p_val = get_mid_p(ma, mb, mc, md)

                        # --- แสดงผลลัพธ์ ---
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

                    except Exception as e:
                        st.error(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ: {e}")
                else:
                    st.warning("กรุณากรอกตัวเลขจำนวนในตาราง 2x2")

    # ------------------------------------------
    # 6.6 Logistic Regression
    # ------------------------------------------
    elif menu == "🧬 Multiple Logistic Regression (AOR)":
        st.title("🧬 Multiple Logistic Regression")
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

            except Exception as e:
                st.error(f"⚠️ ไม่สามารถประมวลผลได้: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #880E4F;'>Epi-Analytic Pro ODPC8 | พัฒนาโดย กลุ่มระบาดวิทยา</div>", unsafe_allow_html=True)
