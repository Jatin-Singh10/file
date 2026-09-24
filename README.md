# 🧠 AI-Based Early Burnout Detection System

An intelligent, early-warning decision-support system designed to estimate the risk of burnout among students. This project leverages a calibrated machine learning model to analyze academic, lifestyle, and psychosocial indicators, providing users with an instant risk assessment and personalized, categorized recommendations.

## 🚀 Key Features

- **Interactive Streamlit Dashboard**: Responsive (desktop and phone) UI with a gradient header, tabbed intake form, a risk gauge with green/amber/red zones, and a downloadable summary to share with a counselor.
- **Explains its estimate**: shows which groups of answers push the score up or down, computed exactly from the logistic-regression coefficients (they sum to the model's log-odds).
- **AI-Powered Risk Prediction**: Uses a trained and calibrated Logistic Regression model to calculate a burnout risk probability.
- **Categorized Recommendation Engine**: Provides tailored advice across four key dimensions:
    - 🚨 **Immediate Support**: High-priority alerts for critical risks.
    - 📚 **Academic Strategies**: Tips for workload and study management.
    - 🌿 **Wellness & Lifestyle**: Guidance on sleep, diet, and balance.
    - 💼 **Professional Guidance**: Pointers toward financial and mental health support.
- **Explainable Logic**: Incorporates "Domain Features" (like Sleep-Work Imbalance and Academic Strain) to capture complex burnout patterns.
- **Critical Alert System**: Triggers immediate emergency warnings for critical indicators (e.g., suicidal thoughts) regardless of the probability score.

## 🛠️ Technical Architecture

### 1. The ML Model
- **Algorithm**: Logistic Regression (selected for its transparency and performance on the specific dataset).
- **Calibration**: Platt-calibrated to ensure the predicted probabilities are reliable.
- **Optimization**: Tuned for **High Recall** to prioritize catching at-risk individuals early.
- **Performance**: High ROC-AUC and PR-AUC, ensuring a strong balance between sensitivity and specificity.

### 2. Risk Assessment Logic
Tiers are anchored to the model's tuned early-warning threshold (`best_threshold` in `metadata.json`, currently 0.22), so a tier and the early-warning alert never contradict each other:

| Risk Tier | Probability Range | Action/Meaning |
| :--- | :--- | :--- |
| **Low** | below the threshold (< 22%) | No early-warning alert; maintain current habits. |
| **Medium** | threshold – 64% | Early-warning alert; monitor indicators and consider a counselor check-in. |
| **High** | 65% – 100% | High risk; strongly recommend professional consultation. |

Tier logic lives in `burnout_logic.py` (`risk_tier`, `HIGH_CUTOFF`).

### 3. Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **ML Library**: Scikit-Learn, Joblib
- **Data Handling**: Pandas, NumPy
- **Deployment**: Docker, Streamlit Community Cloud, Hugging Face Spaces

## 📦 Installation & Usage

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Jatin-Singh10/Burnout-detection.git
   cd Burnout-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard (from inside the project folder):
   ```bash
   streamlit run app.py
   ```
   If the `streamlit` command is not found (common on Windows), use:
   ```bash
   python -m streamlit run app.py
   ```
   Then open http://localhost:8501. Supports Python 3.9+ (tested on 3.12).

### Project layout
`app.py` (UI) · `burnout_logic.py` (features, tiers, recommendations, explanations) · `burnout_ui.py` (HTML/CSS helpers) · `.streamlit/config.toml` (theme) · `burnout_model.joblib` + `metadata.json` (model artifacts) · `test_burnout.py`

### Tests
```bash
pip install pytest
pytest -q
```
Covers feature engineering, tier/alert consistency, the crisis message, recommendation rules, and an end-to-end headless run of the Streamlit app.

### Deployment
The project is ready for deployment via:
- **Streamlit Community Cloud**: Connect your GitHub repo to `share.streamlit.io`.
- **Hugging Face Spaces**: Create a new Space with the Streamlit SDK.
- **Docker**:
  ```bash
  docker build -t burnout-app .
  docker run -p 8501:8501 burnout-app
  ```

## 📌 Known Limitations
- **Proxy target:** the model is trained on the survey's `Depression` label, used here as a proxy for burnout risk. It does not predict clinical burnout or depression diagnoses.
- **Work Pressure / Job Satisfaction are not inputs:** they were almost always 0 in the training data (std ≈ 0.04), so any other value is an extreme extrapolation. They are fixed to 0.
- **Input ranges** follow the training data (`numeric_ranges` in `metadata.json`): Age 18–59, Work/Study Hours 0–12.
- **Explanations are correlations, not causes.** Some factors (e.g. Age) reflect quirks of the survey data.
- **Strong single drivers:** suicidal-thoughts, age, academic pressure, financial stress and diet dominate the model. Age has a large negative effect, likely a dataset artifact, so treat outputs as indicative only.
- Reported test metrics in `metadata.json` come from the original training notebook and were not re-verified here (the dataset is not in this repo).

## ⚠️ Disclaimer
This system is a **research prototype** and a decision-support tool. It is **NOT** a clinical diagnostic tool or a substitute for professional mental health assessment. If you or someone you know is in crisis, please contact a qualified counselor or a local mental health helpline immediately.
