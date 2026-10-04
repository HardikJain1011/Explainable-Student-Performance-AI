# 🎓 Explainable Student Performance AI

> **An end-to-end Explainable AI application that predicts student performance, explains individual predictions with SHAP + LIME, and turns model explanations into actionable what-if scenarios.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20%2B%20LIME-purple)](https://shap.readthedocs.io/)

## 🚀 Why this project stands out

This is not just a classification notebook. It combines **machine learning, model evaluation, explainable AI, counterfactual reasoning, interactive UI and responsible-AI communication** into one deployable application.

A student can:

1. Enter an academic/lifestyle profile.
2. Get a Low / Medium / High performance prediction.
3. See model probabilities instead of only a class label.
4. Understand which features helped or hurt the prediction using SHAP.
5. Cross-check the local explanation with LIME.
6. Receive model-grounded improvement suggestions.
7. Explore what-if combinations that could move the model toward the High class.
8. Download a shareable PDF performance report.

> **Important:** the included dataset is synthetic. The application demonstrates the ML/XAI engineering workflow and should not be interpreted as a validated academic decision system.

---

## ✨ Features

### Student dashboard

- Clean student-facing prediction workflow
- Performance probability chart
- Profile snapshot
- Strongest positive and negative model contributors
- Priority improvement area
- Downloadable PDF report

### Machine learning

- Logistic Regression baseline
- Random Forest
- XGBoost
- Stratified train/test split
- 5-fold cross-validation
- Accuracy, precision, recall, macro-F1 and ROC-AUC
- Automatic best-model selection by macro-F1

### Explainable AI

- Local **SHAP** feature attribution
- Local **LIME** explanation
- Global SHAP summary
- SHAP vs permutation importance
- Feature dependence plots
- Human-readable explanation cards

### Actionable XAI

The project separates **explanation** from **recommendation**.

SHAP answers:

> *Which inputs pushed the current prediction up or down?*

The recommendation layer then looks only at reasonably actionable factors such as attendance, study time, assignment completion, sleep and extracurricular load.

The what-if engine tests plausible combinations of changes and reports whether the trained model would change its predicted class.

This is explicitly presented as a **model simulation, not a causal claim**.

---

## 🧠 Architecture

```text
                    ┌─────────────────────┐
                    │   Student Profile   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Trained ML Models  │
                    │ LR / RF / XGBoost   │
                    └──────────┬──────────┘
                               │
                     Prediction + Probability
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       ┌──────────────┐                  ┌──────────────┐
       │     SHAP     │                  │     LIME     │
       │ local/global │                  │ local check  │
       └──────┬───────┘                  └──────┬───────┘
              └────────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Actionable XAI      │
                    │ Suggestions + What-if│
                    └──────────┬──────────┘
                               ▼
             ┌──────────────────────────────────┐
             │ Streamlit Student Dashboard     │
             │ + Downloadable PDF Report       │
             └──────────────────────────────────┘
```

---

## 📊 Current benchmark

The bundled trained artifacts currently select **XGBoost** by macro-F1.

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.778 | 0.784 | 0.920 |
| Random Forest | 0.762 | 0.767 | 0.907 |
| **XGBoost** | **0.782** | **0.787** | **0.919** |

These numbers are benchmark results on the included synthetic dataset, not evidence of real-world academic prediction accuracy.

---

## 🗂️ Project structure

```text
xai-student-performance/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
│
├── data/
│   └── student_data.csv      # Synthetic dataset
│
├── models/
│   ├── all_models.joblib     # Trained models
│   ├── metrics.csv           # Evaluation metrics
│   ├── meta.json             # Selected model + metadata
│   ├── X_train.csv
│   ├── X_test.csv
│   └── y_test.csv
│
└── src/
    ├── data.py               # Data generation + feature definitions
    ├── train.py              # Training + evaluation pipeline
    ├── explain.py            # SHAP + LIME helpers
    ├── recommendations.py    # Suggestions + counterfactual search
    └── report.py             # PDF report generator
```

---

## ▶️ Run locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd xai-student-performance
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

The app will open in your browser.

### Retrain the models

```bash
python -m src.train
```

The training script regenerates the synthetic data and writes the model artifacts to `models/`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Select your GitHub repository.
4. Set the main file to `app.py`.
5. Deploy.

The repository includes the trained artifacts, so the app can start without retraining. If you intentionally remove the artifacts, `app.py` can regenerate them on first launch.

---

## 🧪 Responsible AI considerations

This project intentionally demonstrates good communication around model explanations:

- **SHAP/LIME are explanations of model behaviour**, not proof of causal relationships.
- Counterfactual suggestions are **what-if simulations**, not guarantees.
- Previous GPA and exam score are treated as explanatory inputs rather than short-term interventions.
- The dataset is synthetic and should not be presented as evidence of real-world student prediction capability.
- A real deployment would require representative consented data, external validation, calibration, fairness testing, privacy controls and model monitoring.

---

## 🔧 Future improvements

- Replace synthetic data with a real, consented educational dataset.
- Add probability calibration and confidence intervals.
- Add fairness evaluation across relevant groups where legally and ethically appropriate.
- Add model/data drift monitoring.
- Add authentication and privacy-preserving storage for a real deployment.
- Add experiment tracking with MLflow or Weights & Biases.
- Add automated tests and CI/CD with GitHub Actions.
- Add a production API layer using FastAPI.

---

## 💼 Resume-ready description

**Explainable Student Performance AI | Python, XGBoost, SHAP, LIME, Streamlit**

- Built an end-to-end multiclass ML application comparing Logistic Regression, Random Forest and XGBoost, achieving **0.787 macro-F1** and **0.919 ROC-AUC** on a held-out synthetic benchmark.
- Implemented **SHAP and LIME** for local/global model interpretability and cross-validated feature importance against permutation importance.
- Developed an actionable XAI layer that converts model attributions into personalized improvement suggestions and **counterfactual what-if simulations**.
- Deployed the workflow as an interactive **Streamlit dashboard** with probability visualizations and downloadable PDF performance reports.

### Short version

> Built an explainable student-performance prediction system using XGBoost, SHAP and LIME, with an interactive Streamlit dashboard, actionable recommendations and counterfactual what-if analysis.

---

## 🧑‍💻 Skills demonstrated

**Machine Learning:** classification, model comparison, cross-validation, evaluation metrics, XGBoost  
**Explainable AI:** SHAP, LIME, permutation importance, counterfactual analysis  
**Python:** pandas, NumPy, scikit-learn, joblib  
**Product/Deployment:** Streamlit, Plotly, PDF generation, Git/GitHub  
**Responsible AI:** limitations, non-causal explanations, synthetic-data disclosure

---

## 📄 License

Add your preferred open-source license before publishing the repository publicly.
