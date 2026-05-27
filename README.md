# Epi-Analytic Pro ODPC8

Streamlit web app สำหรับวิเคราะห์ข้อมูลระบาดวิทยาเชิงพรรณนา, Epi Curve, Spot Map, Bivariate Analysis, และ Multiple Logistic Regression พร้อม AI summary ด้วย Gemini

## โครงสร้างไฟล์สำหรับ GitHub

```text
EpiAnalyticPro_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## รันบนเครื่อง local

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## ตั้งค่า Gemini API Key

### Local
คัดลอก `.streamlit/secrets.toml.example` เป็น `.streamlit/secrets.toml` แล้วใส่ key:

```toml
GEMINI_API_KEY = "your-key"
```

### Streamlit Community Cloud
ไปที่ App settings > Secrets แล้วใส่:

```toml
GEMINI_API_KEY = "your-key"
```

## Deploy ด้วย GitHub + Streamlit Community Cloud

1. สร้าง repository ใหม่ใน GitHub
2. อัปโหลดไฟล์ทั้งหมดในโฟลเดอร์นี้ขึ้น GitHub
3. เข้า Streamlit Community Cloud
4. เลือก New app > เลือก repository > main file path: `app.py`
5. เพิ่ม `GEMINI_API_KEY` ใน Secrets หากต้องใช้ AI summary
6. Deploy

## จุดที่ปรับปรุงจากไฟล์เดิม

- ปรับภาพรวม UI เป็นแนว Modern AI/Data Analytics ด้วย hero section, glass cards, gradient buttons และ metric cards
- เพิ่ม `.streamlit/config.toml` สำหรับ theme กลางของ Streamlit
- เพิ่ม `requirements.txt` สำหรับ deploy ผ่าน GitHub/Streamlit Cloud
- เพิ่มตัวอย่าง secrets management สำหรับ Gemini API Key
- ปรับให้ `generate_ai_summary()` อ่าน API key จาก `st.secrets` ได้ หากไม่ได้กรอกผ่าน sidebar
- เพิ่ม data overview dashboard เมื่ออัปโหลดข้อมูลสำเร็จ
- เก็บฟังก์ชันวิเคราะห์หลักจากไฟล์เดิมไว้
