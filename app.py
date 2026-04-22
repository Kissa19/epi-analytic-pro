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
# 1. CONFIGURATION & Professional Styling
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
    """จัดการรหัส 1/2 หรือ 1.0/2.0 ให้เป็น 1/0 สำหรับการคำนวณทางระบาดวิทยา"""
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
        ["👤 พรรณนา (Descriptive)", 
         "📊 สร้าง Epi Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Bivariate Analysis (OR/RR)", 
         "🧬 Adjusted Analysis (Logistic)",
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
# 6. MAIN CONTENT MODULES
# ==========================================

# --- หน้าลงทะเบียน ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ลงทะเบียนเข้าใช้งานระบบ")
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

    # ------------------------------------------
    # 6.1 Descriptive Analysis
    # ------------------------------------------
    if menu == "👤 พรรณนา (Descriptive)":
        st.title("👤 ระบาดวิทยาเชิงพรรณนา (Descriptive Analysis)")
        st.info(f"📋 จำนวนผู้ป่วยทั้งหมด (n) = {total_n} ราย")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1. เพศ (Sex)")
            sex_col = st.selectbox("เลือกตัวแปรเพศ", df.columns)
            sex_df = df[sex_col].value_counts().reset_index()
            sex_df.columns = ['เพศ', 'จำนวน (n)']
            sex_df['ร้อยละ (%)'] = (sex_df['จำนวน (n)']/total_n*100).round(2)
            st.table(sex_df.style.format({'ร้อยละ (%)': '{:.2f}'}))

        with c2:
            st.subheader("2. อายุ (Age Groups)")
            age_col = st.selectbox("เลือกตัวแปรอายุ", df.columns)
            df['age_grp'] = pd.cut(df[age_col], bins=[0,5,15,25,35,45,55,65,120], 
                                  labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
            age_df = df['age_grp'].value_counts().sort_index().reset_index()
            age_df.columns = ['กลุ่มอายุ', 'จำนวน (n)']
            age_df['ร้อยละ (%)'] = (age_df['จำนวน (n)']/total_n*100).round(2)
            st.table(age_df.style.format({'ร้อยละ (%)': '{:.2f}'}))

        st.subheader("3. อาการและอาการแสดง (1=มีอาการ)")
        symp_cols = st.multiselect("เลือกตัวแปรอาการ", df.columns)
        if symp_cols:
            s_data = [{"อาการ": c, "จำนวน (n)": int((df[c]==1).sum()), "ร้อยละ (%)": ((df[c]==1).sum()/total_n*100)} for c in symp_cols]
            s_df = pd.DataFrame(s_data).sort_values("จำนวน (n)", ascending=True)
            st.table(pd.DataFrame(s_data).sort_values("จำนวน (n)", ascending=False).style.format({'ร้อยละ (%)': '{:.2f}'}))
            fig_s = px.bar(s_df, x="ร้อยละ (%)", y="อาการ", orientation='h', title="ความถี่ของอาการ (ร้อยละ)", text_auto='.1f', color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_s, use_container_width=True)

        st.subheader("4. ปัจจัยเสี่ยง (1=มีปัจจัย)")
        risk_cols = st.multiselect("เลือกตัวแปรปัจจัยเสี่ยง", [c for c in df.columns if c not in symp_cols])
        if risk_cols:
            r_data = [{"ปัจจัย": c, "จำนวน (n)": int((df[c]==1).sum()), "ร้อยละ (%)": ((df[c]==1).sum()/total_n*100)} for c in risk_cols]
            st.table(pd.DataFrame(r_data).sort_values("จำนวน (n)", ascending=False).style.format({'ร้อยละ (%)': '{:.2f}'}))

    # ------------------------------------------
    # 6.2 Epidemic Curve (Bin & Padding)
    # ค้นหาเมนู "📊 สร้าง Epi Curve (Time)" แล้วแทนที่ด้วยโค้ดชุดนี้ครับ
    elif menu == "📊 สร้าง Epi Curve (Time)":
        st.title("📊 Interactive Epidemic Curve (Advanced)")
        date_col = st.sidebar.selectbox("เลือกคอลัมน์วันเริ่มป่วย", df.columns)
        col_grp = st.sidebar.selectbox("ตัวแปรจัดกลุ่ม (Stacked Color):", ["<none>"] + list(df.columns))
        
        unit_map = {"Hour": "H", "Day": "D", "Week": "W", "Month": "M", "30 Min": "30min"}
        bin_unit = st.sidebar.selectbox("หน่วยเวลา (Unit)", list(unit_map.keys()), index=0) 
        bin_size = st.sidebar.number_input("ขนาด Bin", min_value=1, value=1)
        freq = f"{bin_size}{unit_map[bin_unit]}"

        pad_before = st.sidebar.number_input(f"เพิ่มช่องว่างก่อนหน้า ({bin_unit})", value=1)
        pad_after = st.sidebar.number_input(f"เพิ่มช่องว่างข้างหลัง ({bin_unit})", value=1)

        # 1. แปลงรูปแบบวันที่ให้รองรับไฟล์ CSV ของคุณ (วัน/เดือน/ปี ค.ศ. ชั่วโมง:นาที)
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df_clean = df.dropna(subset=[date_col]).copy()

        if not df_clean.empty:
            min_dt = df_clean[date_col].min()
            max_dt = df_clean[date_col].max()
            
            # 2. Logic การตั้งขอบเขตแบบ Dynamic (ถ้าวิเคราะห์รายชั่วโมง/นาที ไม่ต้องถอยไปถึงต้นวัน)
            # --- ส่วนที่ปรับปรุงเพื่อแก้ ValueError ---
        if not df_clean.empty:
            min_dt = df_clean[date_col].min()
            max_dt = df_clean[date_col].max()
            
            # ปรับ Frequency Aliases ให้เป็นตัวพิมพ์เล็กตามมาตรฐาน Pandas 2.x
            if "Hour" in bin_unit or "Min" in bin_unit:
                # ใช้ 'h' สำหรับ Hour และ 'min' สำหรับ Minute
                start_range = (min_dt - pd.Timedelta(hours=pad_before)).floor('h')
                end_range = (max_dt + pd.Timedelta(hours=pad_after)).ceil('h')
            else:
                # ใช้ 'd' สำหรับ Day
                start_range = (min_dt - pd.to_timedelta(pad_before, unit='d')).floor('d')
                end_range = (max_dt + pd.to_timedelta(pad_after, unit='d')).ceil('d')
            
            # สร้าง "ไม้บรรทัดเวลา" (Full Range)
            # หมายเหตุ: freq ใน unit_map ควรเป็นตัวพิมพ์เล็ก เช่น 'h', 'd', 'min'
            full_range = pd.date_range(start=start_range, end=end_range, freq=freq)

            # 3. จัดกลุ่มข้อมูลราย Bin
            if col_grp == "<none>":
                counts = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size()
                chart_df = counts.reindex(full_range, fill_value=0).reset_index()
                chart_df.columns = [date_col, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', text_auto=True, color_discrete_sequence=['#3498db'])
            else:
                counts = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().unstack(fill_value=0)
                chart_df = counts.reindex(full_range, fill_value=0).stack().reset_index(name='Cases')
                chart_df.columns = [date_col, col_grp, 'Cases']
                fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp, color_discrete_sequence=px.colors.qualitative.Alphabet)

            # 4. ปรับการแสดงผลให้เป็นมาตรฐานระบาดวิทยา
            fig.update_layout(
                bargap=0.01, # แท่งชิดกัน
                xaxis_title="Onset Date/Time", 
                yaxis_title="Number of Cases",
                hovermode="x unified",
                xaxis=dict(type='date', tickformat='%d/%m %H:%M') # แสดงเวลาในแกน X
            )
            fig.update_traces(marker_line_width=0.5, marker_line_color='white')
            
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✅ แสดงผลช่วงเวลา {start_range.strftime('%H:%M')} ถึง {end_range.strftime('%H:%M')}")
        else:
            st.error("❌ ไม่สามารถวิเคราะห์ได้ เนื่องจากรูปแบบวันที่ในไฟล์ไม่ถูกต้อง")

    # ------------------------------------------
    # 6.3 Spot Map
    # ------------------------------------------
    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map - GIS Analytics")
        lat_c = next((c for c in df.columns if any(p in c.lower() for p in ['lat', 'latitude', 'ละติจูด'])), None)
        lon_c = next((c for c in df.columns if any(p in c.lower() for p in ['lon', 'longitude', 'ลองจิจูด'])), None)
        
        if lat_c and lon_c:
            df_m = df.dropna(subset=[lat_c, lon_c]).copy()
            m = folium.Map(location=[df_m[lat_c].mean(), df_m[lon_c].mean()], zoom_start=15)
            for idx, r in df_m.iterrows():
                folium.CircleMarker([r[lat_c], r[lon_c]], radius=7, color='red', fill=True).add_to(m)
            folium_static(m, width=1000)
        else: st.warning("⚠️ ไม่พบคอลัมน์พิกัด (Lat/Lon) ในไฟล์")

    # ------------------------------------------
    # 6.4 Bivariate Analysis (OR/RR)
    # ------------------------------------------
    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        st.title("🔬 Bivariate Analysis (OR/RR)")
        out_v = st.selectbox("เลือกตัวแปร Outcome (เช่น ป่วย/ไม่ป่วย)", df.columns)
        exp_list = st.multiselect("เลือกตัวแปรปัจจัยเสี่ยง (Exposures)", [c for c in df.columns if c != out_v])
        
        if st.button("🚀 ประมวลผลสถิติ"):
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

    # ------------------------------------------
    # 6.5 Adjusted Analysis (Logistic)
    # ------------------------------------------
    elif menu == "🧬 Adjusted Analysis (Logistic)":
        st.title("🧬 Multiple Logistic Regression (Adjusted OR)")
        out_v = st.selectbox("Outcome", df.columns, key="log_out")
        exp_v = st.selectbox("ปัจจัยหลัก", [c for c in df.columns if c != out_v])
        adj_v = st.multiselect("ตัวแปรกวน (Adjusted For)", [c for c in df.columns if c not in [out_v, exp_v]])
        
        if st.button("วิเคราะห์ Logistic Regression"):
            try:
                df_m = df[[out_v, exp_v] + adj_v].copy().dropna()
                for c in df_m.columns: df_m[c] = smart_map_variable(df_m[c])
                
                formula = f"Q('{out_v}') ~ Q('{exp_v}')"
                if adj_v: formula += " + " + " + ".join([f"Q('{a}')" for a in adj_v])
                
                model = smf.logit(formula, data=df_m).fit(disp=0)
                res_df = pd.DataFrame({"Adjusted OR": np.exp(model.params), "P-value": model.pvalues})
                st.dataframe(res_df[res_df.index != 'Intercept'].style.format("{:.4f}"))
                st.success("✅ คำนวณ Adjusted OR สำเร็จ")
            except Exception as e: st.error(f"Error: {e}")

else:
    st.info("👈 กรุณานำเข้าไฟล์ข้อมูลดิบที่แถบด้านซ้าย")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)
