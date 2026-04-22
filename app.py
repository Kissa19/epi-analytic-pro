import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import timedelta
import scipy.stats as stats
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
    # พยายามโหลดโลโก้หน่วยงาน
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
        ["👤 พรรณนา (Descriptive)", 
         "📊 สร้าง Epi Curve (Time)", 
         "🗺️ Spot Map (Place)",
         "🔬 Crude Analysis (OR/RR)", 
         "🧬 Adjusted Analysis (Logistic)",
         "📑 สรุปรายงาน & Sensitivity",
         "📈 Dashboard ผู้บริหาร (Researcher)",
         "💬 ข้อเสนอแนะ",
         "📝 ข้อมูลการลงทะเบียน (แก้ไข)"]
    )

# --- Helper Functions ---
def load_data(file):
    try:
        return pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์ได้: {e}")
        return None

def smart_map_variable(series):
    # สำหรับ Logistic Regression (1=Case, 2=Control -> 1, 0)
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({1, 2, 1.0, 2.0}):
        return series.map({1: 1, 2: 0})
    return series

# ==========================================
# 4. MAIN CONTENT AREA
# ==========================================

# --- หน้าลงทะเบียน ---
if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ระบบลงทะเบียนใช้งาน")
    st.markdown("กรุณาระบุข้อมูลเพื่อเข้าถึงระบบวิเคราะห์สถิติระบาดวิทยา")

    with st.form("registration_form"):
        u_name = st.text_input("ชื่อ-นามสกุล", value="" if not st.session_state['registered'] else "ผู้ใช้งานเดิม")
        u_agency = st.text_input("หน่วยงาน / ทีม SRRT-CDCU")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรคหน้างาน", "วิจัย/วิชาการ", "ซ้อมแผนฯ"])
        submit_btn = st.form_submit_button("บันทึกข้อมูลและเริ่มใช้งาน")
        
        if submit_btn and u_name and u_agency:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                new_row = pd.DataFrame([{
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": u_name, "Agency": u_agency, "Purpose": u_purpose
                }])
                existing_df = conn.read(worksheet="Registration")
                conn.update(worksheet="Registration", data=pd.concat([existing_df, new_row], ignore_index=True))
            except: pass
            st.session_state['registered'] = True
            st.balloons()
            st.rerun()

# --- เมนูวิเคราะห์ (ต้องลงทะเบียนก่อน) ---
elif st.session_state['registered']:
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ข้อมูล (Excel/CSV)", type=['xlsx', 'csv'])

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            total_n = len(df)

            # 1. ระบาดวิทยาเชิงพรรณนา
            if menu == "👤 พรรณนา (Descriptive)":
                st.title("👤 ระบาดวิทยาเชิงพรรณนา (Descriptive Analysis)")
                st.info(f"📋 จำนวนผู้ป่วยทั้งหมด (n) = {total_n} ราย")

                # เพศ
                st.subheader("1. เพศ (Sex)")
                sex_col = st.selectbox("เลือกตัวแปรเพศ", df.columns)
                sex_df = df[sex_col].value_counts().reset_index()
                sex_df.columns = ['เพศ', 'จำนวน (n)']
                sex_df['ร้อยละ (%)'] = (sex_df['จำนวน (n)']/total_n*100).round(2)
                st.table(sex_df.style.format({'ร้อยละ (%)': '{:.2f}'}))

                # อายุ
                st.subheader("2. อายุ (Age Groups)")
                age_col = st.selectbox("เลือกตัวแปรอายุ", df.columns)
                df['age_grp'] = pd.cut(df[age_col], bins=[0,5,15,25,35,45,55,65,120], 
                                         labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
                age_df = df['age_grp'].value_counts().sort_index().reset_index()
                age_df.columns = ['กลุ่มอายุ', 'จำนวน (n)']
                age_df['ร้อยละ (%)'] = (age_df['จำนวน (n)']/total_n*100).round(2)
                st.table(age_df.style.format({'ร้อยละ (%)': '{:.2f}'}))

                # อาชีพ
                st.subheader("3. อาชีพ (Occupation)")
                occ_col = st.selectbox("เลือกตัวแปรอาชีพ", df.columns)
                occ_df = df[occ_col].value_counts().reset_index()
                occ_df.columns = ['อาชีพ', 'จำนวน (n)']
                occ_df['ร้อยละ (%)'] = (occ_df['จำนวน (n)']/total_n*100).round(2)
                st.table(occ_df.style.format({'ร้อยละ (%)': '{:.2f}'}))

                # อาการ (Stacked Horizontal Bar)
                st.subheader("4. อาการและอาการแสดง (Symptoms)")
                symp_cols = st.multiselect("เลือกตัวแปรอาการ (คำนวณจากค่า 1=มีอาการ)", df.columns)
                if symp_cols:
                    s_data = [{"อาการ": c, "จำนวน (n)": int((df[c]==1).sum()), "ร้อยละ (%)": ((df[c]==1).sum()/total_n*100)} for c in symp_cols]
                    s_df = pd.DataFrame(s_data).sort_values("จำนวน (n)", ascending=True)
                    st.table(pd.DataFrame(s_data).sort_values("จำนวน (n)", ascending=False).style.format({'ร้อยละ (%)': '{:.2f}'}))
                    fig_s = px.bar(s_df, x="จำนวน (n)", y="อาการ", orientation='h', title="ความถี่ของอาการ")
                    st.plotly_chart(fig_s, use_container_width=True)

                # ปัจจัยเสี่ยง
                st.subheader("5. ปัจจัยเสี่ยง (Risk Factors)")
                risk_cols = st.multiselect("เลือกตัวแปรปัจจัยเสี่ยง (1=มีปัจจัย)", [c for c in df.columns if c not in symp_cols])
                if risk_cols:
                    r_data = [{"ปัจจัย": c, "จำนวน (n)": int((df[c]==1).sum()), "ร้อยละ (%)": ((df[c]==1).sum()/total_n*100)} for c in risk_cols]
                    st.table(pd.DataFrame(r_data).sort_values("จำนวน (n)", ascending=False).style.format({'ร้อยละ (%)': '{:.2f}'}))

            # 2. Epi Curve (Stacked)
      			elif menu == "📊 สร้าง Epi Curve (Time)":
        	st.title("📊 Interactive Epidemic Curve")
        	if uploaded_file:
            	      df = load_data(uploaded_file)
            	      if df is not None:
                	# --- ส่วนการเลือกคอลัมน์และตัวแปรกวน ---
                	date_options = df.columns.tolist()
                	default_index = 0
                	for i, col in enumerate(date_options):
                   	        if any(k in col.lower() for k in ['date', 'onset', 'เริ่ม', 'ป่วย']):
                        	default_index = i
                        	break
                
                # เลือกคอลัมน์วันที่และคอลัมน์สำหรับจัดกลุ่ม (Stacked)
                date_col = st.sidebar.selectbox("เลือกคอลัมน์วันที่เริ่มป่วย", date_options, index=default_index)
                group_options = ["<none>"] + [c for c in df.columns if c != date_col]
                col_grp = st.sidebar.selectbox("ตัวแปรสำหรับจัดกลุ่ม (Stacked Color):", group_options, index=0)
                
                # 1. แปลงข้อมูลวันที่ และจัดการ Error
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                
                # 2. กรองเฉพาะแถวที่มีวันที่ถูกต้อง
                df_clean = df.dropna(subset=[date_col]).copy()

                if df_clean.empty:
                    st.error(f"❌ ไม่พบข้อมูลวันที่ในคอลัมน์ '{date_col}' หรือรูปแบบวันที่ไม่ถูกต้อง")
                    st.info("💡 แนะนำ: ตรวจสอบว่าปีเป็น ค.ศ. และไม่มีค่าว่างในคอลัมน์นี้")
                else:
                    # ตั้งค่าความถี่ของกราฟ
                    unit_map = {"Hour": "H", "Day": "D", "Week": "W", "Month": "M"}
                    c1, c2 = st.sidebar.columns(2)
                    bin_size = c1.number_input("Bin Size", min_value=1, value=1)
                    bin_unit = c2.selectbox("Unit", list(unit_map.keys()), index=1)
                    freq = f"{bin_size}{unit_map[bin_unit]}"

                    # --- 3. การเตรียมข้อมูลสำหรับการวาดกราฟ (Handle Grouping) ---
                    if col_grp == "<none>":
                        # แบบเดี่ยว (ไม่มีกลุ่ม)
                        chart_df = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name='Cases')
                        color_param = None
                        color_seq = ["#89CFF0"] # สีฟ้ามาตรฐาน
                    else:
                        # แบบแยกกลุ่ม (Stacked Bar)
                        chart_df = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().reset_index(name='Cases')
                        color_param = col_grp
                        # ใช้ชุดสี Alphabet หรือ Plotly ที่มีความหลากหลายสูง (คล้าย tab20)
                        color_seq = px.colors.qualitative.Alphabet 

                    # ส่วนขอบเขตการแสดงผล
                    min_date = chart_df[date_col].min().date()
                    max_date = chart_df[date_col].max().date()
                    st.sidebar.subheader("ขอบเขตการแสดงผล")
                    start_p = st.sidebar.date_input("เริ่มจาก", min_date - timedelta(days=2))
                    end_p = st.sidebar.date_input("ถึงวันที่", max_date + timedelta(days=2))
                    
                    # 4. สร้างกราฟด้วย Plotly
                    fig = px.bar(
                        chart_df, 
                        x=date_col, 
                        y='Cases',
                        color=color_param,
                        title=f"Epidemic Curve: {bin_size} {bin_unit} {'(Stacked by ' + col_grp + ')' if col_grp != '<none>' else ''}",
                        color_discrete_sequence=color_seq
                    )
                    
                    fig.update_layout(
                        bargap=0, # ให้แท่งติดกันตามมาตรฐาน Epi Curve
                        xaxis_range=[pd.to_datetime(start_p), pd.to_datetime(end_p) + timedelta(days=1)],
                        xaxis_title="Date of Onset",
                        yaxis_title="Number of Cases",
                        legend_title=col_grp if col_grp != "<none>" else "",
                        hovermode="x unified"
                    )
                    
                    # เพิ่มเส้นขอบสีขาวบางๆ ให้แยกแท่งได้ชัดเจนขึ้นเวลาซ้อนกัน
                    fig.update_traces(marker_line_width=0.5, marker_line_color='white')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ ประมวลผลสำเร็จจากข้อมูลที่มีวันที่สมบูรณ์จำนวน {len(df_clean)} ราย")

            # 3. Spot Map
            elif menu == "🗺️ Spot Map (Place)":
                st.title("🗺️ Spot Map")
                if 'Latitude' in df.columns and 'Longitude' in df.columns:
                    st.map(df.dropna(subset=['Latitude', 'Longitude'])[['Latitude', 'Longitude']])
                else: st.warning("ไม่พบคอลัมน์พิกัดในไฟล์")

            # 4. Crude Analysis
            elif menu == "🔬 Crude Analysis (OR/RR)":
                st.title("🔬 Crude Analysis")
                out_v = st.selectbox("Outcome", df.columns)
                exp_v = st.selectbox("Exposure", df.columns)
                if st.button("คำนวณ"):
                    data = df[[exp_v, out_v]].dropna()
                    data = data[data[exp_v].isin([1, 2]) & data[out_v].isin([1, 2])]
                    a, b, c, d = len(data[(data[exp_v]==1)&(data[out_v]==1)]), len(data[(data[exp_v]==1)&(data[out_v]==2)]), len(data[(data[exp_v]==2)&(data[out_v]==1)]), len(data[(data[exp_v]==2)&(data[out_v]==2)])
                    or_val = (a*d)/(b*c) if (b*c) != 0 else 0
                    st.metric("Crude OR", f"{or_val:.2f}")

            # 5. Adjusted Analysis
            elif menu == "🧬 Adjusted Analysis (Logistic)":
                st.title("🧬 Adjusted Analysis")
                out_v = st.selectbox("Outcome (Target)", df.columns)
                exp_v = st.selectbox("Main Exposure", [c for c in df.columns if c != out_v])
                adj_v = st.multiselect("Covariates", [c for c in df.columns if c not in [out_v, exp_v]])
                if st.button("วิเคราะห์ Logistic"):
                    try:
                        df_m = df[[out_v, exp_v] + adj_v].copy().dropna()
                        for c in df_m.columns: df_m[c] = smart_map_variable(df_m[c])
                        formula = f"{out_v} ~ {exp_v} + {' + '.join(adj_v) if adj_v else '1'}"
                        model = smf.logit(formula, data=df_m).fit(disp=0)
                        res_df = pd.DataFrame({"Adjusted OR": np.exp(model.params), "P-value": model.pvalues})
                        st.dataframe(res_df.style.format("{:.4f}"))
                    except Exception as e: st.error(f"Error: {e}")

            # 6. Report & Dashboard
            elif menu == "📑 สรุปรายงาน & Sensitivity":
                st.title("📑 Report & Sensitivity")
                st.download_button("📥 ดาวน์โหลดรายงาน (.txt)", f"Report n={total_n}", file_name="Report.txt")

            elif menu == "📈 Dashboard ผู้บริหาร (Researcher)":
                st.title("📈 Researcher Dashboard")
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    reg = conn.read(worksheet="Registration")
                    st.metric("ผู้ใช้งานทั้งหมด", len(reg))
                    st.bar_chart(reg['Agency'].value_counts())
                except: st.info("ยังไม่มีข้อมูลการใช้งาน")

            elif menu == "💬 ข้อเสนอแนะ":
                st.title("💬 ข้อเสนอแนะ")
                with st.form("fb_form"):
                    txt = st.text_area("ความเห็นต่อระบบ")
                    if st.form_submit_button("ส่ง"): st.success("ขอบคุณครับ")

    else:
        st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้าย")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยา สคร.8 อุดรธานี</div>", unsafe_allow_html=True)
