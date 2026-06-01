# Final-Project-MachineLearning-EpsilonAI

# 🎓 Student Performance — Machine Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Epsilon AI Machine Learning Program —This project was developed as part of the Epsilon AI program.

Related repository: Epsilon-AI/Main-Repository**

[📊 EDA Notebook](#eda-notebook) • [🤖 ML Notebook](#ml-notebook) • [🖥️ Apps](#streamlit-apps) • [📁 Structure](#repository-structure)

</div>

---

## 📌 Project Overview

This project applies end-to-end machine learning to predict **student academic performance** using behavioral, academic, and demographic data. It covers the complete ML workflow: exploratory data analysis, data cleaning, feature engineering, model training, hyperparameter tuning, and deployment-ready Streamlit applications.

**Two Prediction Tasks:**
| Task | Target | Type | Best Model | Score |
|------|--------|------|------------|-------|
| 🎯 Classification | `Grade` (A/B/C/D/F) | Multi-class | XGBoost | F1 = **90.4%** |
| 📈 Regression | `Final_Score` (0–100) | Continuous | XGBoost | R² = **93.1%** |

---

## 🎯 Problem Statement

Academic institutions struggle to identify at-risk students early enough for effective intervention. Traditional grading lacks predictive power — by the time a student fails, it's often too late.

This project answers:
- Which factors most strongly predict student grade?
- Can we accurately forecast the numeric final exam score?
- What behavioral patterns differentiate high and low performers?
- How can data-driven insights support early intervention?

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| Students | 1,000 records |
| Features | 17 original + 5 engineered |
| Missing Values | ~5% in 3 columns (treated) |
| Duplicates | None found |

### Feature Summary

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Integer | Student age (17–25) |
| `Gender` | Categorical | Male / Female |
| `Major` | Categorical | Engineering, Business, Arts, Science, Medicine |
| `Attendance` | Float | % of classes attended |
| `Hours_Studied` | Float | Weekly study hours |
| `Previous_GPA` | Float | Prior semester GPA (0–4.0) |
| `Sleep_Hours` | Float | Average daily sleep hours |
| `Stress_Level` | Integer | Self-reported stress level (1–10) |
| `Screen_Time` | Float | Daily non-study screen time (hours) |
| `Exam_Anxiety_Score` | Float | Anxiety score before exams (1–10) |
| `Study_Group` | Binary | Participates in study groups (0/1) |
| `Internet_Access` | Binary | Reliable internet access (0/1) |
| `Extracurricular` | Binary | Involved in extracurriculars (0/1) |
| `Family_Income` | Categorical | Low / Medium / High |
| `Teacher_Quality` | Categorical | Low / Medium / High |
| `Final_Score` | Float | **Regression Target** — Final exam score (0–100) |
| `Grade` | Categorical | **Classification Target** — A, B, C, D, F |

---

## 🧹 Data Cleaning

| Issue | Treatment | Rationale |
|-------|-----------|-----------|
| Missing values (~5%) in `Attendance`, `Sleep_Hours`, `Exam_Anxiety_Score` | **Median imputation** | Robust to skewed distributions |
| Duplicate rows | None found | Dataset integrity confirmed |
| Outliers in bounded features | **Winsorization (IQR capping)** | Preserves data volume while reducing extreme influence |
| Data type inconsistencies | Corrected categorical/binary types | Ensures correct pipeline encoding |

**Result:** Clean dataset with 1,000 rows, 17 columns, 0 missing values, 0 duplicates.

---

## 📈 EDA Highlights

Key findings from exploratory data analysis:

1. **Previous GPA** is the strongest single predictor (r = 0.82 with Final Score)
2. **Attendance below 70%** is a strong indicator of Grade D or F
3. **High stress (≥ 8)** correlates with 10–15 point lower scores
4. **Screen time > 6 hrs/day** negatively impacts performance
5. **Study group participation** boosts average score by ~5 points
6. **7–8 hours sleep** is the optimal range for academic performance

---

## ⚙️ Feature Engineering

Five new features were created to improve model signal:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `Study_Efficiency` | `Hours_Studied × (Attendance / 100)` | Effective study time weighted by attendance |
| `Academic_Risk_Index` | `(Stress + Anxiety + Screen_Time) / 3 − GPA × 2` | Single at-risk score for counseling flags |
| `Engagement_Index` | `Attendance×0.4 + Group×20 + Extra×10 + Hours×3` | Holistic academic engagement score |
| `Wellness_Score` | `Sleep×10 − Stress×5 − Anxiety×3` | Physical & mental health composite |
| `Learning_Efficiency` | `Previous_GPA × Hours_Studied / (1 + Screen_Time)` | Quality-adjusted study performance |

---

## 🎯 Feature Selection

Two methods used for robust feature selection:

| Method | Top Features |
|--------|-------------|
| **Mutual Information** | Previous_GPA, Learning_Efficiency, Attendance, Engagement_Index, Hours_Studied |
| **XGBoost Importance** | Previous_GPA, Learning_Efficiency, Attendance, Engagement_Index, Hours_Studied |

**12 features selected** (consistent across both methods).

---

## 🤖 Classification Models — Grade Prediction

**Target:** `Grade` (A, B, C, D, F) — 5-class classification

### Models Trained

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model — fast, interpretable |
| Random Forest Classifier | Ensemble of decision trees — handles non-linearity |
| XGBoost Classifier | Gradient boosting — sequential error correction |

### Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 83.2% | 82.9% | 83.2% | 82.8% |
| Random Forest | 89.1% | 88.7% | 89.1% | 88.8% |
| **⚡ XGBoost** | **90.7%** | **90.3%** | **90.7%** | **90.4%** |

**🏆 Best Classification Model: XGBoost Classifier (F1 = 90.4%)**

---

## 📈 Regression Models — Final Score Prediction

**Target:** `Final_Score` (continuous, 0–100)

### Models Trained

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline — minimizes squared residuals |
| Random Forest Regressor | Bagging ensemble — averages tree predictions |
| XGBoost Regressor | Gradient boosting — minimizes MSE sequentially |

### Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 5.62 | 7.21 | 0.718 |
| Random Forest | 3.41 | 4.68 | 0.912 |
| **🚀 XGBoost** | **3.18** | **4.32** | **0.931** |

**🏆 Best Regression Model: XGBoost Regressor (R² = 93.1%)**

---

## 🔬 Hyperparameter Tuning

**Method:** GridSearchCV with 5-fold Stratified Cross-Validation

Models tuned:
- Random Forest Classifier → Best: `n_estimators=200, max_depth=20`
- XGBoost Classifier → Best: `n_estimators=200, learning_rate=0.1`
- Random Forest Regressor → Best: `n_estimators=200, max_depth=20`
- XGBoost Regressor → Best: `n_estimators=200, learning_rate=0.05`

**Average improvement after tuning: +0.7–0.9% F1/R²**

---

## ✅ Validation

| Technique | Purpose |
|-----------|---------|
| **80/20 Train-Test Split** | Stratified split preserving class balance |
| **5-Fold Cross-Validation** | Reliable performance estimate, detects overfitting |

Both XGBoost models showed low cross-validation variance (± < 1.5%), confirming good generalization.

---

## 🖥️ Streamlit Applications

### App 1 — EDA Dashboard (`streamlit_eda_app.py`)

Business-facing analytics dashboard for non-technical users:
- 📊 Interactive grade & score distribution charts
- 🎛️ Filters by Grade, Gender, Major, Attendance range
- 📈 Feature vs Final Score scatter plots
- 🌡️ Correlation heatmap
- 👥 Demographic analysis (Gender, Major, Family Income)
- 💡 6 key business insight cards
- 📋 Color-coded raw data table

**Run:** `streamlit run streamlit_eda_app.py`

---

### App 2 — ML Prediction App (`streamlit_ml_app.py`)

Real-time student performance predictor:
- 🎛️ Sliders for all student features
- 🔮 Instant Grade & Score predictions (XGBoost)
- 📊 Score gauge visualization (F–A range)
- 💪 Strengths & improvement recommendations
- ⚙️ 5 engineered feature metrics
- 📈 Profile vs class average comparison

**Run:** `streamlit run streamlit_ml_app.py`

---

## 📁 Repository Structure

```
student-performance-ml/
│
├── EDA_Analysis.ipynb          ← Exploratory Data Analysis (10 sections)
├── Machine_Learning.ipynb      ← ML models, tuning, evaluation (7 sections)
├── streamlit_eda_app.py        ← Business EDA Dashboard
├── streamlit_ml_app.py         ← ML Prediction Application
├── presentation.pptx           ← 20-slide professional presentation
├── README.md                   ← This file
└── screenshots/
    ├── eda_dashboard.png       ← EDA app screenshot
    └── ml_predictor.png        ← ML app screenshot
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/student-performance-ml.git
cd student-performance-ml

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost streamlit jupyter

# 4. Run EDA Notebook
jupyter notebook EDA_Analysis.ipynb

# 5. Run ML Notebook
jupyter notebook Machine_Learning.ipynb

# 6. Launch EDA Dashboard
streamlit run streamlit_eda_app.py

# 7. Launch ML Prediction App
streamlit run streamlit_ml_app.py
```

---

## 📦 Requirements

```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
xgboost>=1.5
streamlit>=1.15
jupyter>=1.0
scipy>=1.7
```

---

## 🔭 Future Improvements

| Improvement | Description |
|-------------|-------------|
| Deep Learning | Add Neural Network baseline (MLP) for comparison |
| Time-Series Analysis | Track performance across multiple semesters |
| SHAP Explainability | Individual prediction explanations using SHAP values |
| Cloud Deployment | Deploy to Streamlit Cloud / Hugging Face Spaces |
| Live Data Pipeline | Real-time ingestion from student information systems |
| A/B Testing | Measure impact of interventions suggested by the model |
| Multi-School Dataset | Generalize across different institutions |
| Mobile App | React Native mobile interface for educators |

---

## 👤 Author

**Student Name**  
Epsilon AI Machine Learning Program  
GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgment

This project was completed as part of the **Epsilon AI Machine Learning Program**.

Special thanks to **Epsilon AI** for providing the learning materials, guidance, and project framework that made this work possible.

📎 Official Epsilon AI GitHub Repository: [github.com/epsilon-ai](https://github.com/epsilon-ai)

The structured curriculum, hands-on project approach, and mentor support from Epsilon AI were instrumental in producing a university-quality machine learning project from data to deployment.

---

## 📄 License

This project is for educational purposes as part of the Epsilon AI ML Program.

---

<div align="center">

Made with ❤️ using Python, Scikit-Learn, XGBoost & Streamlit

**Epsilon AI Machine Learning Program**

</div>
