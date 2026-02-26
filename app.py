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
    st.title("📝 ระบบลงทะเบียนใช้งาน (PDPA Compliant)")
    st.info("ℹ️ ระบบจะบันทึกข้อมูลเฉพาะประเภททีมและหน่วยงานเพื่อใช้ในการสรุปผลเชิงสถิติเท่านั้น โดยไม่เก็บข้อมูลส่วนบุคคล")

    with st.form("registration_form"):
        # 1. ประเภททีม (ตามที่คุณปรับแก้โค้ดไว้)
        u_team_type = st.selectbox("ประเภททีม", ["CDCU", "SRRT", "SAT", "JIT", "อื่นๆ"])
        
        # 2. หน่วยงาน
        u_agency = st.text_input("หน่วยงาน / สังกัด")
        
        # 3. วัตถุประสงค์
        u_purpose = st.selectbox("วัตถุประสงค์การใช้งาน", [
            "สอบสวนโรคภาคสนาม (Field Investigation)", 
            "วิเคราะห์ข้อมูลทางระบาดวิทยา", 
            "ซ้อมแผนตอบโต้ภาวะฉุกเฉิน", 
            "การเรียนการสอน/วิชาการ"
        ])

        if st.form_submit_button("บันทึกข้อมูลและเริ่มใช้งาน"):
            # ดึงเวลาปัจจุบัน
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # เตรียมข้อมูล (ตัดชื่อ-นามสกุลออกแล้ว)
            new_reg_entry = {
                "วันที่-เวลา": current_time,
                "ประเภททีม": u_team_type,
                "หน่วยงาน": u_agency,
                "วัตถุประสงค์การใช้งาน": u_purpose
            }

            try:
                # เชื่อมต่อกับ Google Sheets ตามลิงก์ที่คุณให้มา
                log_url = "https://docs.google.com/spreadsheets/d/1YRqCkU2JABM61evfyPJny3tYzAIwuJFCRVAJ-uuKrHE/edit?usp=sharing"
                conn = st.connection("gsheets", type=GSheetsConnection)
                
                # อ่านข้อมูลเดิม (ถ้ามี)
                existing_data = conn.read(spreadsheet=log_url)
                
                # ต่อท้ายข้อมูลใหม่
                updated_df = pd.concat([existing_data, pd.DataFrame([new_reg_entry])], ignore_index=True)
                
                # อัปเดตกลับไปยัง Google Sheets
                conn.update(spreadsheet=log_url, data=updated_df)
                
                # เก็บสถานะลงทะเบียนในเครื่องผู้ใช้
                st.session_state['registered'] = True
                st.success("✅ ลงทะเบียนสำเร็จ และเริ่มระบบวิเคราะห์ข้อมูล")
                st.balloons()
                st.rerun()

            except Exception as e:
                # กรณีบันทึกลง Sheet ไม่สำเร็จ (เช่น สิทธิ์การเขียน) ยังคงให้ใช้งานแอปต่อได้
                st.session_state['registered'] = True
                st.warning(f"⚠️ ระบบบันทึกสถิติขัดข้อง แต่ท่านสามารถใช้งานแอปได้ปกติ: {e}")

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

    # 5. Spot Map
    elif menu == "🗺️ Spot Map (Place)":
        st.title("🗺️ Spot Map & Buffer Zone")
        lat_col = next((c for c in df.columns if c.lower() in ['latitude', 'lat']), None)
        lon_col = next((c for c in df.columns if c.lower() in ['longitude', 'lon']), None)
        if lat_col and lon_col:
            radius_m = st.sidebar.selectbox("รัศมี Buffer:", [0, 100, 200, 500, 1000], index=3)
            df_map = df.dropna(subset=[lat_col, lon_col]).copy()
            if not df_map.empty:
                import folium
                from streamlit_folium import folium_static
                m = folium.Map(location=[df_map[lat_col].mean(), df_map[lon_col].mean()], zoom_start=14)
                for idx, row in df_map.iterrows():
                    folium.CircleMarker([row[lat_col], row[lon_col]], radius=5, color='red', fill=True).add_to(m)
                    if radius_m > 0: folium.Circle([row[lat_col], row[lon_col]], radius=radius_m, color='blue', fill=True, fill_opacity=0.1, weight=1).add_to(m)
                folium_static(m, width=1200, height=600)
            else: st.error("พิกัดไม่ถูกต้อง")

elif st.session_state['registered'] and df is None:
    st.info("👈 กรุณาอัปโหลดไฟล์ หรือวางลิงก์ Google Sheets เพื่อเริ่มการวิเคราะห์")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("---")

st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข สคร.8 อุดรธานี</div>", unsafe_allow_html=True)








