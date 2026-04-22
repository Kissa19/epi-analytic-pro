Python
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
    """แปลงค่า 1,2 ให้เป็น 1,0 อัตโนมัติ (1->1, 2->0)"""
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({1, 2, 1.0, 2.0}):
        return series.map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    return series

def calculate_mid_p(a, b, c, d):
    """คำนวณ Mid-P Exact P-value แบบ Epi Info"""
    n = a + b + c + d
    if n == 0: return np.nan
    k = a + c # total cases
    m = a + b # total exposed
    p_obs = hypergeom.pmf(a, n, k, m)
    # 2-tailed Mid-P = 2 * min(P(X < a) + 0.5P(X=a), P(X > a) + 0.5P(X=a))
    p_lower = hypergeom.cdf(a - 1, n, k, m) + 0.5 * p_obs
    p_upper = (1 - hypergeom.cdf(a, n, k, m)) + 0.5 * p_obs
    mid_p = 2 * min(p_lower, p_upper)
    return min(mid_p, 1.0)

# ==========================================
# 4. MAIN CONTENT AREA
# ==========================================

if menu == "📝 ลงทะเบียนใช้งาน" or menu == "📝 ข้อมูลการลงทะเบียน (แก้ไข)":
    st.title("📝 ระบบลงทะเบียนใช้งาน")
    with st.form("reg_form"):
        u_name = st.text_input("ชื่อ-นามสกุล", value="" if not st.session_state['registered'] else "ผู้ใช้งานเดิม")
        u_agency = st.text_input("หน่วยงาน / ทีม SRRT-CDCU")
        u_purpose = st.selectbox("วัตถุประสงค์", ["สอบสวนโรคหน้างาน", "วิจัย/วิชาการ", "ซ้อมแผนฯ"])
        if st.form_submit_button("บันทึกข้อมูลและเริ่มใช้งาน"):
            st.session_state['registered'] = True
            st.balloons()
            st.rerun()

elif st.session_state['registered']:
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ข้อมูล (Excel/CSV)", type=['xlsx', 'csv'])

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            total_n = len(df)

            if menu == "👤 พรรณนา (Descriptive)":
                st.title("👤 ระบาดวิทยาเชิงพรรณนา")
                st.info(f"📋 จำนวนผู้ป่วยทั้งหมด (n) = {total_n} ราย")
                
                # ตัวแปรพื้นฐาน
                for label, col_key in [("1. เพศ (Sex)", "sex"), ("2. อายุ (Age Group)", "age"), ("3. อาชีพ (Occupation)", "occ")]:
                    st.subheader(label)
                    sel_col = st.selectbox(f"เลือกตัวแปร {label}", df.columns, key=col_key)
                    if label == "2. อายุ (Age Group)":
                        df['age_group'] = pd.cut(df[sel_col], bins=[0,5,15,25,35,45,55,65,120], labels=['0-4','5-14','15-24','25-34','35-44','45-54','55-64','65+'])
                        res = df['age_group'].value_counts().sort_index().reset_index()
                    else:
                        res = df[sel_col].value_counts().reset_index()
                    res.columns = ['รายการ', 'จำนวน (n)']
                    res['ร้อยละ (%)'] = (res['จำนวน (n)']/total_n*100).round(2)
                    st.table(res.style.format({'ร้อยละ (%)': '{:.2f}'}))

                # อาการ (Horizontal Bar Chart)
                st.subheader("4. อาการและอาการแสดง (Symptoms)")
                symp_cols = st.multiselect("เลือกตัวแปรอาการ (1=มีอาการ)", df.columns)
                if symp_cols:
                    s_data = [{"อาการ": c, "จำนวน (n)": int((df[c]==1).sum()), "ร้อยละ (%)": ((df[c]==1).sum()/total_n*100)} for c in symp_cols]
                    s_df = pd.DataFrame(s_data).sort_values("จำนวน (n)", ascending=True)
                    st.table(pd.DataFrame(s_data).sort_values("จำนวน (n)", ascending=False).style.format({'ร้อยละ (%)': '{:.2f}'}))
                    st.plotly_chart(px.bar(s_df, x="จำนวน (n)", y="อาการ", orientation='h', title="ความถี่ของอาการ"), use_container_width=True)
 				
		# ปัจจัยเสี่ยง
                st.subheader("5. ปัจจัยเสี่ยง (Risk Factors)")
                risk_cols = st.multiselect("เลือกตัวแปรปัจจัยเสี่ยง (1=มีปัจจัย)", [c for c in df.columns if c not in symp_cols])
                if risk_cols:
                    r_data = [{"ปัจจัย": c, "จำนวน (n)": int((df[c]==1).sum()), "ร้อยละ (%)": ((df[c]==1).sum()/total_n*100)} for c in risk_cols]
                    st.table(pd.DataFrame(r_data).sort_values("จำนวน (n)", ascending=False).style.format({'ร้อยละ (%)': '{:.2f}'}))


            elif menu == "📊 สร้าง Epi Curve (Time)":
                st.title("📊 Epidemic Curve Creat")
                
                # --- ส่วนควบคุมใน Sidebar ---
                st.sidebar.subheader("ตั้งค่าแกน X และการแยกสี")
                date_options = df.columns.tolist()
                date_col = st.sidebar.selectbox("เลือกวันที่เริ่มป่วย", date_options)
                
                group_options = ["<none>"] + [c for c in df.columns if c != date_col]
                col_grp = st.sidebar.selectbox("ตัวแปรสำหรับแยกสี (Category):", group_options, index=0)
                
                st.sidebar.subheader("ตั้งค่าช่วงเวลา (Bin & Padding)")
                unit_map = {"Hour": "H", "Day": "D", "Week": "W", "Month": "M"}
                c1, c2 = st.sidebar.columns(2)
                bin_size = c1.number_input("ขนาด Bin", min_value=1, value=1)
                bin_unit = c2.selectbox("หน่วย", list(unit_map.keys()), index=1)
                freq = f"{bin_size}{unit_map[bin_unit]}"

                pad_before = st.sidebar.number_input(f"เพิ่มช่วงก่อนพบรายแรก ({bin_unit})", min_value=0, value=1)
                pad_after = st.sidebar.number_input(f"เพิ่มช่วงหลังรายสุดท้าย ({bin_unit})", min_value=0, value=1)

                # --- การจัดการข้อมูล ---
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                df_clean = df.dropna(subset=[date_col]).copy()

                if not df_clean.empty:
                    # คำนวณขอบเขตเวลา
                    min_dt = df_clean[date_col].min()
                    max_dt = df_clean[date_col].max()
                    
                    # คำนวณระยะ Padding
                    u_delta = "hours" if bin_unit == "Hour" else "days" if bin_unit == "Day" else "weeks" if bin_unit == "Week" else "days"
                    val_before = pad_before if bin_unit != "Month" else pad_before * 30
                    val_after = pad_after if bin_unit != "Month" else pad_after * 30
                    
                    start_range = min_dt - pd.Timedelta(**{u_delta: val_before * bin_size})
                    end_range = max_dt + pd.Timedelta(**{u_delta: val_after * bin_size})

                    # --- การเตรียมข้อมูลสำหรับกราฟ ---
                    if col_grp == "<none>":
                        # กรณีไม่แบ่งกลุ่ม: ใช้สีน้ำเงินอ่อนเป็นค่าเริ่มต้น
                        chart_df = df_clean.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name='Cases')
                        fig = px.bar(chart_df, x=date_col, y='Cases', color_discrete_sequence=["#ADD8E6"]) # LightBlue
                    else:
                        # กรณีแบ่งกลุ่ม: แปลงกลุ่มเป็น String เพื่อให้โชว์เลขจำนวนเต็ม และใช้สีแบบ Category
                        df_clean[col_grp] = df_clean[col_grp].apply(lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x))
                        
                        chart_df = df_clean.groupby([pd.Grouper(key=date_col, freq=freq), col_grp]).size().reset_index(name='Cases')
                        # ใช้สีแบบ Category (Qualitative Colors)
                        fig = px.bar(chart_df, x=date_col, y='Cases', color=col_grp, 
                                     color_discrete_sequence=px.colors.qualitative.Set1)

                    # ปรับแต่ง Layout ให้เป็นมาตรฐานระบาดวิทยา
                    fig.update_layout(
                        bargap=0, # แท่งชิดกันตามหลัก Epi Curve
                        xaxis_range=[start_range, end_range],
                        xaxis_title="วันที่เริ่มป่วย (Onset Date)",
                        yaxis_title="จำนวนผู้ป่วย (Cases)",
                        legend_title=f"กลุ่ม: {col_grp}",
                        hovermode="x unified"
                    )
                    # เพิ่มเส้นขอบสีขาวบางๆ เพื่อให้แยกแท่งใน Stacked Bar ได้ชัดเจน
                    fig.update_traces(marker_line_width=0.5, marker_line_color='white')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"📈 แสดงข้อมูลผู้ป่วย {len(df_clean)} ราย (รายแรก: {min_dt.date()} | รายสุดท้าย: {max_dt.date()})")
                else:
                    st.error("ไม่พบข้อมูลวันที่ที่ถูกต้องในคอลัมน์ที่เลือก")

	        # 3. Crude Analysis (แบบละเอียดเหมือน Epi Info)
            elif menu == "🔬 Crude Analysis (OR/RR)":
                st.title("🔬 Binary Analysis: Crude Analysis (Tables)")
                st.markdown("การวิเคราะห์หาความสัมพันธ์รายปัจจัยแบบ 2x2 Table")
                
                # การตั้งค่าวิเคราะห์
                c1, c2 = st.columns(2)
                with c1:
                    out_v = st.selectbox("เลือกตัวแปรตาม (Outcome: ป่วย=1, ไม่ป่วย=0)", df.columns)
                with c2:
                    design = st.radio("Study Design", ["Case-control (Odds Ratio)", "Cohort (Risk Ratio)"])
                
                exp_list = st.multiselect("เลือกปัจจัยเสี่ยง (Exposures: มี=1, ไม่มี=0)", [c for c in df.columns if c != out_v])
                
                if st.button("🚀 ประมวลผล Crude Analysis"):
                    results = []
                    for exp_v in exp_list:
                        # เตรียมข้อมูลและ Auto-map 1,2 เป็น 1,0
                        temp = df[[out_v, exp_v]].copy().dropna()
                        temp[out_v] = smart_map_variable(temp[out_v])
                        temp[exp_v] = smart_map_variable(temp[exp_v])
                        
                        # กรองเอาเฉพาะ 1 และ 0
                        temp = temp[temp[out_v].isin([1, 0]) & temp[exp_v].isin([1, 0])]
                        
                        if len(temp) > 0:
                            # a = ป่วย(+), ปัจจัย(+) | b = ไม่ป่วย(-), ปัจจัย(+) 
                            # c = ป่วย(+), ปัจจัย(-) | d = ไม่ป่วย(-), ปัจจัย(-)
                            a = len(temp[(temp[exp_v]==1) & (temp[out_v]==1)])
                            b = len(temp[(temp[exp_v]==1) & (temp[out_v]==0)])
                            c = len(temp[(temp[exp_v]==0) & (temp[out_v]==1)])
                            d = len(temp[(temp[exp_v]==0) & (temp[out_v]==0)])
                            
                            # คำนวณ OR หรือ RR
                            try:
                                if "Case-control" in design:
                                    m_label = "Odds Ratio"
                                    val = (a * d) / (b * c) if (b * c) != 0 else np.nan
                                    se = np.sqrt(1/a + 1/b + 1/c + 1/d) if all([a,b,c,d]) else np.nan
                                else:
                                    m_label = "Risk Ratio"
                                    val = (a / (a + b)) / (c / (c + d)) if (a+b)!=0 and (c+d)!=0 and c!=0 else np.nan
                                    se = np.sqrt((1/a - 1/(a+b)) + (1/c - 1/(c+d))) if all([a,c]) and (a+b)!=0 and (c+d)!=0 else np.nan
                                
                                ci_l = np.exp(np.log(val) - 1.96 * se) if not np.isnan(val) else np.nan
                                ci_u = np.exp(np.log(val) + 1.96 * se) if not np.isnan(val) else np.nan
                                p_val = calculate_mid_p(a, b, c, d)
                                
                                results.append({
                                    "ปัจจัย (Exposure)": exp_v,
                                    "ป่วย (+)": a, "ไม่ป่วย (+)": b, 
                                    "ป่วย (-)": c, "ไม่ป่วย (-)": d,
                                    m_label: val,
                                    "95% CI Lower": ci_l,
                                    "95% CI Upper": ci_u,
                                    "P-value (Mid-P)": p_val
                                })
                            except: pass

                    if results:
                        st.subheader(f"📊 ตารางสรุปผล ({m_label})")
                        res_df = pd.DataFrame(results)
                        st.dataframe(res_df.style.format({
                            m_label: "{:.2f}", "95% CI Lower": "{:.2f}", 
                            "95% CI Upper": "{:.2f}", "P-value (Mid-P)": "{:.4f}"
                        }).apply(lambda x: ['background-color: #e8f5e9' if x['P-value (Mid-P)'] < 0.05 else '' for _ in x], axis=1), use_container_width=True)
                    else:
                        st.warning("ไม่พบข้อมูลที่เพียงพอสำหรับการวิเคราะห์ (ตรวจสอบว่าข้อมูลเป็น 1/0 หรือ 1/2)")

            # --- 4. Adjusted Analysis (Logistic) ---
            elif menu == "🧬 Adjusted Analysis (Logistic)":
                st.title("🧬 Adjusted Analysis: Multiple Logistic Regression")
                out_v = st.selectbox("เลือกตัวแปรตาม (Outcome: 1=ป่วย, 0=ไม่ป่วย)", df.columns)
                exp_v = st.selectbox("เลือกตัวแปรอิสระหลัก", [c for c in df.columns if c != out_v])
                adj_v = st.multiselect("เลือกตัวแปรกวน (Covariates)", [c for c in df.columns if c not in [out_v, exp_v]])
                
                if st.button("🚀 ประมวลผลแบบละเอียด"):
                    try:
                        df_m = df[[out_v, exp_v] + adj_v].copy().dropna()
                        df_m[out_v] = smart_map_variable(df_m[out_v])
                        df_m[exp_v] = smart_map_variable(df_m[exp_v])
                        for c in adj_v: df_m[c] = smart_map_variable(df_m[c])
                        
                        formula = f"{out_v} ~ {exp_v} + {' + '.join(adj_v) if adj_v else '1'}"
                        model = smf.logit(formula, data=df_m).fit(disp=0)
                        
                        summary_df = pd.DataFrame({
                            "Factors": model.params.index,
                            "Adjusted OR": np.exp(model.params.values),
                            "95% CI Lower": np.exp(model.conf_int()[0].values),
                            "95% CI Upper": np.exp(model.conf_int()[1].values),
                            "P-value": model.pvalues.values
                        })
                        st.dataframe(summary_df.style.format("{:.4f}").apply(lambda x: ['background-color: #f1f8e9' if x['P-value'] < 0.05 else '' for _ in x], axis=1))
                    except Exception as e: st.error(f"Error: {e}")

    else: st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้าย")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>Epi-Analytic Pro: พัฒนาโดย กลุ่มระบาดวิทยา สคร.8 อุดรธานี</div>", unsafe_allow_html=True)
