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

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'registered' not in st.session_state:
    st.session_state['registered'] = False

# ==========================================
# 3. SIDEBAR NAVIGATION & LOGO
# ==========================================
try:
    st.sidebar.image("สำนักงานป้องกันควบคุมโรคที่8.png", width=150)
except:
    st.sidebar.title("🏥 ODPC8")

st.sidebar.title("🏥 Epi-Analytic Menu")

if not st.session_state['registered']:
    menu = "📝 ลงทะเบียนใช้งาน"
    st.sidebar.warning("⚠️ โปรดลงทะเบียนเพื่อปลดล็อกเมนูวิเคราะห์")
else:
    menu = st.sidebar.radio(
        "เลือกหัวข้อการวิเคราะห์", 
        ["👤 บุคคล (Person)", 
         "📊 Epidemic Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Bivariate Analysis (OR/RR)", 
         "🧬 Multiple Logistic Regression (Adjusted)",
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
                    
    # 3. Crude Analysis
    elif menu == "🔬 Bivariate Analysis (OR/RR)":
        st.title("🔬 Bivariate Analysis (1=Yes, 0=No)")
        out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns)
        design = st.radio("ประเภทการศึกษา", ["Case-control Study (OR)", "Cohort Study (RR)"])
        exp_list = st.multiselect("เลือกปัจจัยเสี่ยง", [c for c in df.columns if c != out_v])
        
        if st.button("🚀 ประมวลผล"):
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
                        m_label, measure = ("OR", (a*d)/(b*c)) if "Case-control" in design else ("RR", (a/(a+b))/(c/(c+d)))
                        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
                        ci_l, ci_u = np.exp(np.log(measure)-1.96*se), np.exp(np.log(measure)+1.96*se)
                        p_val = calculate_mid_p(a, b, c, d)
                        results.append({"ปัจจัย": exp_v, "ป่วย(+)": a, "ไม่ป่วย(+)": b, "ป่วย(-)": c, "ไม่ป่วย(-)": d, m_label: measure, "95% CI Lower": ci_l, "95% CI Upper": ci_u, "Mid-P": p_val})
                    except: pass
            if results:
                st.dataframe(pd.DataFrame(results).style.format({m_label: "{:.2f}", "95% CI Lower": "{:.2f}", "95% CI Upper": "{:.2f}", "Mid-P": "{:.4f}"}))

    # 4. Adjusted Analysis
    elif menu == "🧬 Multiple Logistic Regression (Adjusted)":
        st.title("🧬 Multiple Logistic Regression (Adjusted)")
        st.markdown("วิเคราะห์ปัจจัยเสี่ยงโดยควบคุมตัวแปรกวน (แสดงค่า AOR และ 95% CI)")
        
        out_v = st.selectbox("ตัวแปรตาม (Outcome)", df.columns, key="log_out")
        exp_v = st.selectbox("ปัจจัยหลัก (Exposure)", [c for c in df.columns if c != out_v], key="log_exp")
        adj_v = st.multiselect("ตัวแปรกวน (Covariates)", [c for c in df.columns if c not in [out_v, exp_v]], key="log_adj")
        
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
        st.title("🗺️ Spot Map - Multi-Dimensional Tracking")
        
        # 1. ค้นหาคอลัมน์สำคัญ
        def find_col(possible_names):
            return next((c for c in df.columns if any(p in c.lower() for p in possible_names)), None)

        lat_c = find_col(['latitude', 'lat', 'ละติจูด'])
        lon_c = find_col(['longitude', 'lon', 'ลองจิจูด'])
        age_c = find_col(['age', 'อายุ'])
        sex_c = find_col(['sex', 'gender', 'เพศ'])
        date_c = find_col(['onset', 'วันที่เริ่มป่วย', 'วันที่ป่วย'])

        if lat_c and lon_c:
            # ล้างข้อมูลพิกัด
            df[lat_c] = pd.to_numeric(df[lat_c], errors='coerce')
            df[lon_c] = pd.to_numeric(df[lon_c], errors='coerce')
            df_m = df.dropna(subset=[lat_c, lon_c]).copy()

            # --- ส่วนจัดการตัวแปร "เพศ" และ "กลุ่มอายุ" ให้เป็นหมวดหมู่ ---
            if sex_c:
                df_m['เพศ (กลุ่ม)'] = df_m[sex_c].astype(str).str.strip().replace({'M': 'ชาย', 'F': 'หญิง', 'male': 'ชาย', 'female': 'หญิง'})
            
            if age_c:
                # สร้างกลุ่มอายุอัตโนมัติ
                def age_grouping(a):
                    try:
                        val = float(str(a).replace(' ปี', '').strip())
                        return "< 15 ปี" if val < 15 else "15 ปีขึ้นไป"
                    except: return "ไม่ระบุ"
                df_m['กลุ่มอายุ'] = df_m[age_c].apply(age_grouping)

            # --- Sidebar Settings ---
            st.sidebar.subheader("⚙️ Map Display Settings")
            map_choice = st.sidebar.radio("รูปแบบแผนที่:", ["Google Hybrid (ดาวเทียม)", "Google Roadmap", "OpenStreetMap"])
            
            # รวมตัวแปรทั้งหมดมาเป็นตัวเลือกเดียว
            # ตัดคอลัมน์ระบบและพิกัดออกเพื่อให้เลือกง่าย
            excluded_cols = [lat_c, lon_c]
            base_options = [c for c in df_m.columns if c not in excluded_cols]
            
            color_by = st.sidebar.selectbox("เลือกตัวแปรเพื่อแยกกลุ่มสี:", ["<สีแดงทั้งหมด>"] + base_options)
            rad = st.sidebar.selectbox("รัศมี Buffer (เมตร):", [0, 100, 200, 500], index=1)

            # --- 2. สร้าง Legend และกำหนดสี (Unified Logic) ---
            legend_dict = {}
            palette = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#42D4F4', '#F032E6', '#BFEF45', '#FABEBE', '#469990', '#E6BEFF', '#9A6324', '#FFFAC8', '#800000', '#AAFFC3', '#808000', '#FFD8B1', '#000075', '#A9A9A9']
            
            if color_by == "<สีแดงทั้งหมด>":
                legend_dict = {f"ผู้ป่วยทั้งหมด (n={len(df_m)})": "red"}
            else:
                # นับจำนวนเคสในแต่ละกลุ่มเพื่อแสดงใน Legend
                counts = df_m[color_by].value_counts()
                unique_vals = counts.index.tolist()
                
                # กำหนดสีให้แต่ละกลุ่มค่า
                for i, val in enumerate(unique_vals):
                    color = palette[i % len(palette)]
                    legend_dict[f"{val} (n={counts[val]})"] = color

            # --- 3. สร้างแผนที่ ---
            tiles_url = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}' if "Hybrid" in map_choice else \
                        'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}' if "Roadmap" in map_choice else 'OpenStreetMap'
            
            m = folium.Map(location=[df_m[lat_c].mean(), df_m[lon_c].mean()], zoom_start=16, tiles=tiles_url, attr='Google')

            # วาดจุดข้อมูล
            for idx, r in df_m.iterrows():
                # หาค่าสีจากคำอธิบายใน Legend (ตัดส่วน n=... ออกเพื่อหา Key)
                if color_by == "<สีแดงทั้งหมด>":
                    dot_color = "red"
                else:
                    val = str(r[color_by])
                    # ค้นหาสีที่ตรงกับค่านั้นๆ
                    dot_color = next((v for k, v in legend_dict.items() if k.startswith(val)), "gray")

                # Popup ข้อมูล
                pop_content = f"<b>ลำดับเคส: {idx+1}</b><br>{color_by}: {r.get(color_by, 'N/A')}"
                
                folium.CircleMarker(
                    location=[r[lat_c], r[lon_c]],
                    radius=8, color=dot_color, fill=True, fill_opacity=0.8,
                    tooltip=f"เคสที่ {idx+1}",
                    popup=folium.Popup(pop_content, max_width=200)
                ).add_to(m)

                if rad > 0:
                    folium.Circle([r[lat_c], r[lon_c]], radius=rad, color='blue', fill=True, fill_opacity=0.05, weight=1).add_to(m)

            # แสดงผลแผนที่
            folium_static(m, width=1000, height=600)

            # --- 4. แสดง Legend ใน Sidebar พร้อมจำนวนเคส ---
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"📍 สัญลักษณ์: {color_by}")
            
            legend_html = "<div style='background-color:#ffffff; padding:10px; border-radius:8px; border:1px solid #ccc;'>"
            for label, color in legend_dict.items():
                legend_html += f"""
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <div style='width:14px; height:14px; background-color:{color}; border-radius:50%; margin-right:10px; border:1px solid #333;'></div>
                    <span style='font-size:13px; font-weight:500;'>{label}</span>
                </div>
                """
            legend_html += "</div>"
            st.sidebar.markdown(legend_html, unsafe_allow_html=True)
            
        else:
            st.error("⚠️ ไม่พบคอลัมน์พิกัด Lat/Lon ในไฟล์ของคุณ")
# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("---")

st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)



















