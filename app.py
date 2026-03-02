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

import streamlit as st

# เพิ่ม CSS เพื่อปรับแต่ง Sidebar
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

        /* 4. ปรับแต่งให้โลโก้ดูคมชัด (ลดเงาหรือฟิลเตอร์ที่อาจรบกวน) */
        [data-testid="stSidebar"] img {
            object-fit: contain;
            image-rendering: -webkit-optimize-contrast; /* ช่วยให้ภาพคมชัดขึ้นในบาง Browser */
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
# แสดงโลโก้ สคร.8 (ใช้ไฟล์ที่คุณอัปโหลดขึ้นมา)
try:
    # แนะนำให้ตรวจสอบว่าชื่อไฟล์ตรงกับที่อัปโหลดมาเป๊ะๆ (รวมสระและนามสกุล)
    st.sidebar.image("odpc8_logo.png", use_container_width=True)
except:
    # หากหาไฟล์ไม่เจอ ให้โชว์ไอคอนแทน
    st.sidebar.title("🏥 ODPC8 Udon Thani")

st.sidebar.markdown("---")

st.sidebar.title("🏥 Epi-Analytic Menu")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนเพื่อปลดล็อกเมนูวิเคราะห์ 🚫 คำเตือนความปลอดภัย: โปรดตรวจสอบไฟล์ Excel และลบคอลัมน์ ชื่อ-นามสกุล หรือข้อมูลระบุตัวตนออกก่อนอัปโหลดเข้าสู่ระบบทุกครั้ง")
else:
    menu = st.sidebar.radio(
        "เลือกหัวข้อการวิเคราะห์", 
        ["👥 ประชากรและอัตราป่วย (Attack Rate)",
         "👤 บุคคล (Person)", 
         "📊 Epidemic Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Bivariate Analysis (OR/RR)", 
         "🧬 Multiple Logistic Regression (Adjusted OR; AOR)",
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
    """แปลงค่า 1,2 ให้เป็น 1,0 อัตโนมัติ (1->1, 2->0)"""
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({1, 2, 1.0, 2.0}):
        return series.map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    return series

def calculate_mid_p(a, b, c, d):
    """คำนวณ Mid-P Exact P-value แบบ Epi Info"""
    n = a + b + c + d
    if n == 0: return np.nan
    k = a + c # total sick
    m = a + b # total exposed
    p_obs = hypergeom.pmf(a, n, k, m)
    p_lower = hypergeom.cdf(a - 1, n, k, m) + 0.5 * p_obs
    p_upper = (1 - hypergeom.cdf(a, n, k, m)) + 0.5 * p_obs
    mid_p = 2 * min(p_lower, p_upper)
    return min(mid_p, 1.0)

# ==========================================
# 4. MAIN CONTENT AREA (DATA LOADING)
# ==========================================

# ส่วนการเลือกแหล่งข้อมูล (เพิ่มเข้ามาใหม่เพื่อเชื่อม GSheets)
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
                # เชื่อมต่อ Google Sheets แบบ Real-time
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(spreadsheet=sheet_url)
                if st.sidebar.button("🔄 อัปเดตข้อมูล (Refresh)"):
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"ไม่สามารถเชื่อมต่อ Google Sheets ได้: {e}")
                st.info("💡 แนะนำ: ตรวจสอบว่าแชร์ไฟล์เป็น 'Anyone with the link can view' หรือยัง")

# --- ส่วนหน้าลงทะเบียน (Registration Page) ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
    st.caption("ระบบบันทึกข้อมูลตามมาตรฐาน PDPA ไม่มีการเก็บชื่อ-นามสกุลของผู้ใช้งาน")

    with st.form("reg_form_v2"):
        u_team = st.selectbox("ประเภททีม", ["CDCU", "SRRT", "SAT", "JIT", "อื่นๆ"])
        u_agency = st.text_input("หน่วยงาน / สังกัด (เช่น สสจ.อุดรธานี, รพ.เลย)")
        u_purpose = st.selectbox("วัตถุประสงค์", [
            "สอบสวนโรคภาคสนาม", "วิเคราะห์สถิติวิชาการ", "ซ้อมแผนฯ", "อื่นๆ"
        ])
        
        submit_reg = st.form_submit_button("เริ่มใช้งานระบบ")

        if submit_reg:
            if not u_agency:
                st.error("กรุณาระบุหน่วยงานก่อนเข้าใช้งาน")
            else:
                from datetime import datetime
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # เตรียมข้อมูลสำหรับส่งไปยัง Apps Script
                payload = {
                    "timestamp": now,
                    "team": u_team,
                    "agency": u_agency,
                    "purpose": u_purpose
                }

                try:
                    # 🔴 สำคัญ: เปลี่ยน URL ด้านล่างนี้เป็น Web App URL ที่คุณได้จากการ Deploy ใน Google Sheets
                    url = "https://script.google.com/macros/s/AKfycbxVGzrB9IjdvD90g2Zm8cKNwYE1PMrtaaun7YlBkGjWoL3UjVw74K49B_wg4cBfedeB/exec"
                    
                    # ส่งข้อมูลแบบ POST
                    response = requests.post(url, json=payload)
                    
                    if response.status_code == 200:
                        st.session_state['registered'] = True
                        st.success("✅ บันทึกประวัติการเข้าใช้งานเรียบร้อย")
                        st.balloons()
                        st.rerun()
                    else:
                        raise Exception("เซิร์ฟเวอร์ตอบกลับด้วยสถานะอื่น")
                        
                except Exception as e:
                    # หากบันทึกไม่ได้ (เช่น ลิงก์เสีย) ยังคงอนุญาตให้เข้าใช้งานแอปได้
                    st.session_state['registered'] = True 
                    st.warning(f"⚠️ บันทึกสถิติไม่สำเร็จ แต่ท่านสามารถใช้งานแอปได้ปกติ: {e}")

# --- เมนูวิเคราะห์ ---
elif st.session_state['registered'] and df is not None:
    total_n = len(df)

    # --- เมนูใหม่: ประชากรและ Attack Rate ---
    if menu == "👥 ประชากรและอัตราป่วย (Attack Rate)":
        st.title("👥 ประชากรและอัตราป่วย (Attack Rate)")
        st.info("กรุณาระบุจำนวนประชากรกลุ่มเสี่ยง (Population at Risk) เพื่อคำนวณอัตราป่วย")

        # 1. ค้นหาคอลัมน์อัตโนมัติ
        def find_col(possible_names):
            return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)
        
        sex_c = find_col(['sex', 'gender', 'เพศ'])
        age_c = find_col(['age', 'อายุ'])

        # 2. ส่วนการกรอกข้อมูลประชากร (Denominator)
        st.subheader("1. ระบุจำนวนประชากรกลุ่มเสี่ยงแยกตามกลุ่ม")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("**แยกตามเพศ**")
            pop_male = st.number_input("จำนวนประชากรชายทั้งหมด", min_value=1, value=100)
            pop_female = st.number_input("จำนวนประชากรหญิงทั้งหมด", min_value=1, value=100)
            total_pop_sex = pop_male + pop_female
            st.write(f"รวมประชากร (เพศ): {total_pop_sex:,}")

        with col_p2:
            st.markdown("**แยกตามกลุ่มอายุ (มาตรฐาน)**")
            # สร้าง List กลุ่มอายุมาตรฐาน
            age_labels = ['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+']
            pop_age = {}
            for label in age_labels:
                pop_age[label] = st.number_input(f"ประชากรกลุ่มอายุ {label}", min_value=0, value=0)
            total_pop_age = sum(pop_age.values())
            st.write(f"รวมประชากร (อายุ): {total_pop_age:,}")

        # 3. คำนวณ Attack Rate
        if st.button("📈 คำนวณ Attack Rate"):
            st.markdown("---")
            st.subheader("📊 ผลการคำนวณ Attack Rate")
            
            # --- Overall Attack Rate ---
            overall_ar = (len(df) / total_pop_sex * 100) if total_pop_sex > 0 else 0
            st.metric("Overall Attack Rate (อัตราป่วยรวม)", f"{overall_ar:.2f} %", f"Cases: {len(df)}")

            res_col1, res_col2 = st.columns(2)
            
            # --- Sex-Specific Attack Rate ---
            with res_col1:
                st.markdown("**Sex-Specific Attack Rate**")
                if sex_c:
                    # ปรับรหัสเพศให้เป็น ชาย/หญิง ก่อนนับ
                    df['sex_temp'] = df[sex_c].astype(str).str.strip().replace({'1':'ชาย','2':'หญิง','1.0':'ชาย','2.0':'หญิง'})
                    male_cases = len(df[df['sex_temp'] == 'ชาย'])
                    female_cases = len(df[df['sex_temp'] == 'หญิง'])
                    
                    ar_male = (male_cases / pop_male * 100)
                    ar_female = (female_cases / pop_female * 100)
                    
                    ar_sex_df = pd.DataFrame({
                        "เพศ": ["ชาย", "หญิง"],
                        "จำนวนป่วย (n)": [male_cases, female_cases],
                        "ประชากร (N)": [pop_male, pop_female],
                        "Attack Rate (%)": [ar_male, ar_female]
                    })
                    st.table(ar_sex_df.style.format({"Attack Rate (%)": "{:.2f}"}))
                else:
                    st.warning("ไม่พบคอลัมน์เพศในไฟล์ข้อมูล")

            # --- Age-Specific Attack Rate ---
            with res_col2:
                st.markdown("**Age-Specific Attack Rate**")
                if age_c:
                    # จัดกลุ่มอายุใน df ให้ตรงกับ labels
                    df['age_grp_temp'] = pd.cut(df[age_c], bins=[0,5,15,25,35,45,55,65,120], labels=age_labels, right=False)
                    age_cases = df['age_grp_temp'].value_counts().reindex(age_labels, fill_value=0)
                    
                    age_ar_data = []
                    for label in age_labels:
                        cases = age_cases[label]
                        pop = pop_age[label]
                        ar = (cases / pop * 100) if pop > 0 else 0
                        age_ar_data.append({"กลุ่มอายุ": label, "จำนวนป่วย (n)": cases, "ประชากร (N)": pop, "Attack Rate (%)": ar})
                    
                    st.table(pd.DataFrame(age_ar_data).style.format({"Attack Rate (%)": "{:.2f}"}))
                else:
                    st.warning("ไม่พบคอลัมน์อายุในไฟล์ข้อมูล")
            
            # กราฟเปรียบเทียบ Attack Rate
            st.subheader("🖼️ Attack Rate Comparison")
            fig_ar = px.bar(pd.DataFrame(age_ar_data), x="กลุ่มอายุ", y="Attack Rate (%)", 
                           title="Age-Specific Attack Rate (%)", text_auto='.2f')
            st.plotly_chart(fig_ar, use_container_width=True)
    
    # 1. พรรณนา (Descriptive)
    if menu == "👤 บุคคล (Person)":
        st.title("👤 การกระจายตามบุคคล")
        st.info(f"📋 จำนวนข้อมูลทั้งหมด (n) = {total_n} ราย")
        
        for label, col_key in [("1. เพศ", "sex_s"), ("2. อายุ", "age_s"), ("3. อาชีพ/ชั้นเรียน", "occ_s")]:
            st.subheader(label)
            sel_col = st.selectbox(f"เลือกตัวแปร {label}", df.columns, key=col_key)
            if "อายุ" in label:
                df['age_group'] = pd.cut(df[sel_col], bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
                res = df['age_group'].value_counts().sort_index().reset_index()
            else:
                res = df[sel_col].value_counts().reset_index()
            res.columns = ['รายการ', 'จำนวน (n)']
            res['ร้อยละ (%)'] = (res['จำนวน (n)']/total_n*100).round(2)
            st.table(res.style.format({'ร้อยละ (%)': '{:.2f}'}))

        st.subheader("4. อาการและอาการแสดง (1=มี, 0=ไม่มี)")
        sym_cols = st.multiselect("เลือกตัวแปรอาการ", [c for c in df.columns if c not in ['sex', 'age', 'occ']])
        if sym_cols:
            s_data = []
            for c in sym_cols:
                n_case = int((df[c] == 1).sum())
                pct = (n_case / total_n * 100)
                s_data.append({"อาการ": c, "จำนวน (n)": n_case, "ร้อยละ (%)": pct})
            
            s_df = pd.DataFrame(s_data).sort_values("จำนวน (n)", ascending=True)
            fig_s = px.bar(
                s_df, x="ร้อยละ (%)", y="อาการ", orientation='h', 
                title="ร้อยละอาการและอาการแสดง (ร้อยละ)",
                text=s_df["ร้อยละ (%)"].apply(lambda x: f'{x:.1f}%'), 
                color_discrete_sequence=["#3498db"]
            )
            fig_s.update_traces(textposition='outside')
            fig_s.update_layout(xaxis=dict(range=[0, 110]), xaxis_title="ร้อยละ (%)", yaxis_title="")
            st.plotly_chart(fig_s, use_container_width=True)

        st.subheader("5. ปัจจัย (Factors)")
        risk_cols = st.multiselect("เลือกตัวแปรปัจจัย (1=มีปัจจัย)", [c for c in df.columns if c not in sym_cols])
        if risk_cols:
            r_data = [{"ปัจจัย": c, "จำนวน (n)": int((df[c]==1).sum()), "ร้อยละ (%)": ((df[c]==1).sum()/total_n*100)} for c in risk_cols]
            st.table(pd.DataFrame(r_data).sort_values("จำนวน (n)", ascending=False).style.format({'ร้อยละ (%)': '{:.2f}'}))

    # 2. Epi Curve
    elif menu == "📊 Epidemic Curve (Time)":
                st.title("📊 Interactive Epidemic Curve (ชั่วโมง/วัน)")
                date_col = st.sidebar.selectbox("เลือกตัวแปรเวลา (Onset Date/Time)", df.columns)
                col_grp = st.sidebar.selectbox("ตัวแปรแยกสี (Category):", ["<none>"] + df.columns.tolist())
                
                unit_map = {"Hour": "H", "Day": "D", "Week": "W", "Month": "M"}
                bin_size = st.sidebar.number_input("ขนาด Bin", min_value=1, value=1)
                bin_unit = st.sidebar.selectbox("หน่วย", list(unit_map.keys()), index=0) # Default เป็น Hour สำหรับเคสนี้
                freq = f"{bin_size}{unit_map[bin_unit]}"

                pad_before = st.sidebar.number_input(f"เพิ่มช่วงก่อนหน้า ({bin_unit})", value=2) # เพิ่มเป็น 2 เพื่อความสวยงาม
                pad_after = st.sidebar.number_input(f"เพิ่มช่วงข้างหลัง ({bin_unit})", value=2)

                # แปลงข้อมูลเป็น DateTime
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                df_clean = df.dropna(subset=[date_col]).copy()

                if not df_clean.empty:
                    # --- ส่วนสำคัญ: จัดการเรื่อง Padding ให้ลงล็อกชั่วโมง ---
                    min_dt = df_clean[date_col].min()
                    max_dt = df_clean[date_col].max()

                    if bin_unit == "Hour":
                        # ปัดเศษให้เป็นต้นชั่วโมง (เช่น 14:30 -> 14:00)
                        start_range = (min_dt - pd.Timedelta(hours=pad_before * bin_size)).floor('H')
                        end_range = (max_dt + pd.Timedelta(hours=pad_after * bin_size)).ceil('H')
                    else:
                        # สำหรับรายวัน/สัปดาห์/เดือน
                        offset = pd.to_timedelta(pad_before * bin_size, unit=bin_unit[0].lower()) if bin_unit != "Month" else pd.DateOffset(months=pad_before)
                        start_range = (min_dt - offset).floor('D')
                        end_range = (max_dt + offset).ceil('D')

                    # 1. สร้าง "ไม้บรรทัด" เวลาที่ครบทุกชั่วโมง
                    full_range = pd.date_range(start=start_range, end=end_range, freq=freq)

                    # 2. ประมวลผลข้อมูล
                    if col_grp == "<none>":
                        # นับจำนวนรายชั่วโมง และ Reindex เพื่อเติม 0 ในชั่วโมงที่ไม่มีคนป่วย
                        counts = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size()
                        chart_df = counts.reindex(full_range, fill_value=0).reset_index()
                        chart_df.columns = [date_col, 'Cases']
                        
                        fig = px.bar(chart_df, x=date_col, y='Cases', 
                                     color_discrete_sequence=["#ADD8E6"],
                                     text_auto=True) # โชว์ตัวเลขบนแท่ง
                    else:
                        # กรณีแยกสี (เช่น แยกกลุ่มอาการ หรือสถานที่)
                        counts = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().unstack(fill_value=0)
                        chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name='Cases')
                        chart_df.columns = [date_col, col_grp, 'Cases']
                        
                        fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp,
                                     color_discrete_sequence=px.colors.qualitative.Set1)

                    # 3. ตกแต่งกราฟ
                    fig.update_layout(
                        bargap=0.05, # เพิ่มช่องว่างระหว่างแท่งนิดหน่อยให้ดูง่ายในรายชั่วโมง
                        xaxis_range=[start_range, end_range],
                        xaxis_title="วัน-เวลาที่เริ่มป่วย (Onset Time)",
                        yaxis_title="จำนวนผู้ป่วย (Cases)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"✅ แสดง Epidemic Curve ราย {bin_unit} เรียบร้อยแล้ว (รวม Padding ก่อน-หลัง)")

    # 3. Crude Analysis (Bivariate + Manual 2x2)
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
   
    # 4. Adjusted Analysis
    elif menu == "🧬 Multiple Logistic Regression (Adjusted OR; AOR)":
        st.title("🧬 Multiple Logistic Regression (Adjusted OR; AOR)")
        st.markdown("วิเคราะห์ปัจจัยเสี่ยงโดยควบคุมตัวแปรกวน (แสดงค่า AOR และ 95% CI)")
        
        out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns, key="log_out")
        exp_v = st.selectbox("ปัจจัยหลัก (Exposure)", [c for c in df.columns if c != out_v], key="log_exp")
        adj_v = st.multiselect("ตัวแปรกวน (Confounding)", [c for c in df.columns if c not in [out_v, exp_v]], key="log_adj")
        
        if st.button("🚀 ประมวลผล Logistic Regression"):
            try:
                # 1. เตรียมข้อมูลและทำความสะอาด (จัดการรหัส 1/2 เป็น 1/0)
                cols_needed = [out_v, exp_v] + adj_v
                df_m = df[cols_needed].copy().dropna()
                for col in df_m.columns:
                    df_m[col] = smart_map_variable(df_m[col])
                
                # 2. สร้างสูตรการคำนวณ
                formula = f"Q('{out_v}') ~ Q('{exp_v}')"
                if adj_v:
                    formula += " + " + " + ".join([f"Q('{a}')" for a in adj_v])
                
                # 3. รัน Model
                model = smf.logit(formula, data=df_m).fit(disp=0)
                
                # 4. คำนวณค่า AOR และ 95% CI
                # ใช้ np.exp เพื่อแปลงค่า Coefficient (Log odds) เป็น Odds Ratio
                conf_int = model.conf_int() # ได้ค่า CI ในรูปแบบ Log odds
                
                res_df = pd.DataFrame({
                    "Factors": model.params.index,
                    "Adjusted OR (AOR)": np.exp(model.params.values),
                    "95% CI Lower": np.exp(conf_int[0].values),
                    "95% CI Upper": np.exp(conf_int[1].values),
                    "P-value": model.pvalues.values
                })
                
                # ลบ Intercept และล้างชื่อตัวแปรให้สวยงาม
                res_df = res_df[res_df['Factors'] != 'Intercept']
                res_df['Factors'] = res_df['Factors'].str.extract(r"Q\('(.*)'\)")[0].fillna(res_df['Factors'])
                
                # 5. แสดงผลตาราง
                st.subheader("📋 สรุปผลการวิเคราะห์ปัจจัยเสี่ยงโดยควบคุมตัวแปรกวน")
                st.dataframe(res_df.style.format({
                    "Adjusted OR (AOR)": "{:.2f}",
                    "95% CI Lower": "{:.2f}",
                    "95% CI Upper": "{:.2f}",
                    "P-value": "{:.4f}"
                }).apply(lambda x: ['background-color: #e8f5e9' if x['P-value'] < 0.05 else '' for _ in x], axis=1), 
                use_container_width=True)
                
                st.success("✅ คำนวณค่า Adjusted OR และ 95% CI สำเร็จ")
                
            except Exception as e:
                st.error(f"⚠️ ไม่สามารถประมวลผลได้: {e}")
                st.info("คำแนะนำ: ตรวจสอบว่าตัวแปรอิสระมีจำนวนผู้ป่วย (Case) เพียงพอในแต่ละกลุ่มหรือไม่")

    # 5. Spot Map (ฉบับอัปเกรด Google Maps / Satellite)
    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map - Professional Analytics")
        st.markdown("---")

        # 1. ฟังก์ชันค้นหาคอลัมน์อัตโนมัติ
        def find_col(possible_names):
            return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)

        lat_c = find_col(['latitude', 'lat', 'ละติจูด'])
        lon_c = find_col(['longitude', 'lon', 'ลองจิจูด'])
        age_c = find_col(['age', 'อายุ'])
        sex_c = find_col(['sex', 'gender', 'เพศ'])
        date_c = find_col(['onset', 'วันที่เริ่มป่วย', 'วันที่ป่วย'])

        if lat_c and lon_c:
            # --- เตรียมข้อมูล (Data Cleaning) ---
            df[lat_c] = pd.to_numeric(df[lat_c], errors='coerce')
            df[lon_c] = pd.to_numeric(df[lon_c], errors='coerce')
            df_m = df.dropna(subset=[lat_c, lon_c]).copy()

            # --- จัดกลุ่มข้อมูล เพศ และ อายุ ให้เป็นชื่อใหม่ ---
            if sex_c:
                df_m['เพศ (สรุป)'] = df_m[sex_c].astype(str).str.strip().replace({
                    '1': 'ชาย', '1.0': 'ชาย', 'M': 'ชาย', 'male': 'ชาย',
                    '2': 'หญิง', '2.0': 'หญิง', 'F': 'หญิง', 'female': 'หญิง'
                })
            
            if age_c:
                def categorize_age(x):
                    try:
                        v = float(str(x).replace(' ปี', '').strip())
                        return "< 15 ปี" if v < 15 else "15 ปีขึ้นไป"
                    except: return "ไม่ระบุอายุ"
                df_m['กลุ่มอายุ (สรุป)'] = df_m[age_c].apply(categorize_age)

            # --- ส่วนตั้งค่าใน Sidebar ---
            st.sidebar.subheader("⚙️ ตั้งค่าแผนที่")
            map_choice = st.sidebar.radio("รูปแบบพื้นหลัง:", ["Google Hybrid (ดาวเทียม)", "Google Roadmap", "OpenStreetMap"])
            
            # รวมคอลัมน์ทั้งหมดเพื่อเป็นตัวเลือกการแยกสี
            display_cols = [c for c in df_m.columns if c not in [lat_c, lon_c]]
            color_by = st.sidebar.selectbox("ตัวแปรที่ใช้แยกสีจุด:", ["<สีแดงทั้งหมด>"] + display_cols)
            rad = st.sidebar.selectbox("รัศมีวงรอบ (เมตร):", [0, 100, 200, 500], index=1)

            # --- 2. สร้าง Palette สี และ Legend ---
            legend_dict = {}
            palette = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#42D4F4', '#F032E6', '#BFEF45', '#FABEBE', '#469990', '#E6BEFF', '#9A6324']
            
            if color_by == "<สีแดงทั้งหมด>":
                legend_dict = {f"ผู้ป่วยทั้งหมด (n={len(df_m)})": "red"}
            else:
                counts = df_m[color_by].value_counts()
                for i, (val, count) in enumerate(counts.items()):
                    color = palette[i % len(palette)]
                    legend_dict[f"{val} (n={count})"] = color

            # --- 3. เริ่มสร้างแผนที่ ---
            tiles_url = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}' if "Hybrid" in map_choice else \
                        'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}' if "Roadmap" in map_choice else 'OpenStreetMap'
            
            m = folium.Map(
                location=[df_m[lat_c].mean(), df_m[lon_c].mean()], 
                zoom_start=16, 
                tiles=tiles_url, 
                attr='Google'
            )

            # วาดจุด Marker
            for idx, r in df_m.iterrows():
                # หาความหมายสีจาก Legend
                if color_by == "<สีแดงทั้งหมด>":
                    dot_color = "red"
                else:
                    curr_val = str(r[color_by])
                    dot_color = next((v for k, v in legend_dict.items() if k.startswith(curr_val)), "gray")

                # Popup รายละเอียด
                case_no = idx + 1
                pop_html = f"""
                <div style='font-family: sans-serif; font-size: 12px;'>
                    <b style='color:red;'>📍 ผู้ป่วยรายที่: {case_no}</b><br>
                    <b>{color_by}:</b> {r.get(color_by, 'N/A')}<br>
                    <b>พิกัด:</b> {r[lat_c]}, {r[lon_c]}
                </div>
                """

                folium.CircleMarker(
                    location=[r[lat_c], r[lon_c]],
                    radius=9,
                    color=dot_color,
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=f"เคสที่ {case_no}",
                    popup=folium.Popup(pop_html, max_width=250)
                ).add_to(m)

                if rad > 0:
                    folium.Circle([r[lat_c], r[lon_c]], radius=rad, color='blue', fill=True, fill_opacity=0.05, weight=1).add_to(m)

            # แสดงผลแผนที่
            from streamlit_folium import folium_static
            folium_static(m, width=1000, height=600)

            # --- 4. แสดง Legend ใน Sidebar (ฉบับ Clean HTML) ---
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"📍 สัญลักษณ์: {color_by}")
            
            leg_items_html = ""
            for label, color in legend_dict.items():
                leg_items_html += f"""
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 16px; height: 16px; background-color: {color}; border-radius: 50%; margin-right: 12px; border: 1px solid #333;"></div>
                    <span style="font-size: 14px; font-weight: 500; color: #333;">{label}</span>
                </div>
                """
            
            st.sidebar.markdown(
                f"""
                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);">
                    {leg_items_html}
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.sidebar.caption("💡 คลิกที่จุดเพื่อดูรายละเอียดรายบุคคล")

        else:
            st.error("⚠️ ไม่พบคอลัมน์ละติจูด/ลองจิจูด กรุณาตรวจสอบชื่อคอลัมน์ในไฟล์ Excel (เช่น lat, lon, latitude, longitude)")

# 5. FOOTER
# ==========================================
st.markdown("---")

st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)












































