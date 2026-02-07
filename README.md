🧿 Retina-AI — Clinical Diabetic Retinopathy Screening (MVP)

Retina-AI is an AI-assisted clinical screening MVP for diabetic retinopathy (DR) using retinal fundus images.
It provides end-to-end workflow from patient registry → AI screening → explainability → clinical PDF reports.

⚠️ Decision support only. Final diagnosis must always be made by qualified clinicians.

🚀 Live Demo

• Streamlit App
👉 https://retina-ai-zpkddbsb6m2rf6tfgd6rjh.streamlit.app

• GitHub Repository
👉 https://github.com/Gujjar-Pranav/retina-ai

✨ Core Capabilities
1. Registry

  -  Patient creation & update
    
  -  Diabetes duration, HbA1c, hypertension capture
    
  -  Clinician management

2. Screening

  -  Retinal fundus upload
    
  -  AI inference (DR / No-DR)
    
  -  Confidence + image quality gates
    
  -  Automatic risk stratification
    
  -  Grad-CAM explainability
    
  -  Clinical recommendations

3. Reports

  -  One-page clinical PDF generation
    
  -  Includes:
    
   - Patient summary
    
   - Prediction + confidence
    
   - Risk factors
    
   - Image quality metrics
    
   - Grad-CAM visualization
    
   - Clinician notes

4. Authentication & Roles

  - Login system
    
  -  Role-based access:
    
  -  Admin
    
  -  Registry
    
  -  Screening
    
  -  Reports

5. DevOps

  -  GitHub Actions CI
    
  -  Ruff linting
    
  -  Import smoke tests
    
  -  Streamlit Cloud deployment

🧠 Model

  -  PyTorch binary classifier (DR / No-DR)
    
  -  Grad-CAM explainability
    
  -  CPU / CUDA / Apple MPS supported

🏗 Architecture
Diagram
  flowchart TD
    U[Clinician / Staff / Admin] --> ST[Streamlit UI]

    ST --> AUTH[Auth + Roles]
    AUTH --> TABS[Registry / Screening / Reports]

    TABS --> REG[Registry UI]
    REG --> PX[data/patients.xlsx]
    REG --> DX[data/doctors.xlsx]

    TABS --> SCR[Screening UI]
    SCR --> ML[Model Loader]
    ML --> MODEL[PyTorch Model]

    SCR --> CORE[screening_core]
    CORE --> PRED[Prediction]
    CORE --> RISK[Risk Stratification]
    CORE --> CAM[Grad-CAM]

    SCR --> PDF[pdf_report]
    PDF --> OUT[reports/*.pdf]

    TABS --> REP[Reports Tab]
    REP --> OUT

🗂 Project Structure

app/

 - streamlit_app.py → Main Streamlit entry

src/

 -  ui_registry.py → Registry UI
  
 -  ui_screening.py → Screening workflow
  
 -  pdf_report.py → PDF generator
  
 -  screening_core.py → Model inference + risk logic
  
 -  model_loader.py → PyTorch loader
  
 -  auth.py → Authentication & roles

data/

 -  patients.xlsx → Patient registry
  
 -  doctors.xlsx → Clinician registry

reports/

 -  Generated PDF reports
  
 -  requirements.txt
  
 -  Python dependencies
  
 -  .github/workflows/
  
 -  ci.yml → CI pipeline

🛠 Tech Stack

 - Python 3.10

 - Streamlit

 - PyTorch

 - OpenCV / Pillow

 - Pandas / NumPy

 - ReportLab (PDF)

 - PyMuPDF (preview)

 - Ruff (linting)

 - GitHub Actions (CI)

🧪 Local Setup
1. Clone
- git clone https://github.com/Gujjar-Pranav/retina-ai.git
- cd retina-ai

2. Virtual Environment
- python -m venv .venv
- source .venv/bin/activate

3. Install
- pip install -r requirements.txt

4. Run
- streamlit run app/streamlit_app.py

✅ CI Pipeline

- Triggered on every push:

- Install dependencies

- Ruff lint checks

- Import smoke tests

- Defined in:

- .github/workflows/ci.yml

📌 Notes

- Patient data stored locally in Excel

- Reports saved under /reports

- Streamlit Cloud filesystem is ephemeral

- Model loaded via model_loader.py

📜 License

MIT License

👤 Author

Pranav Gujjar

⚠️ Medical Disclaimer

This software is intended for research and educational purposes only.
It must NOT be used as a standalone diagnostic system.
