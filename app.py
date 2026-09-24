"""
AI-Based Early Burnout Detection System for Students - Streamlit dashboard.
Loads the trained, calibrated model and serves live risk estimates with an
explanation of what drives them and categorized recommendations.

Run:  pip install -r requirements.txt  &&  python -m streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Student Burnout Risk Detector", page_icon="🧠", layout="centered")

try:
    from burnout_logic import (
        CRISIS_MESSAGE, STUDENT_FIXED_INPUTS, add_domain_features,
        describe_factor, explain_factors, recommendation_engine, risk_tier,
    )
    from burnout_ui import CSS, factor_rows_html, gauge_svg, hero_html, tier_html
except ImportError:
    st.error(
        "**Missing file:** `burnout_logic.py` or `burnout_ui.py`. Keep all project files "
        "(`app.py`, `burnout_logic.py`, `burnout_ui.py`, `burnout_model.joblib`, `metadata.json`) "
        "in the same folder."
    )
    st.stop()

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "burnout_model.joblib"
META_PATH = APP_DIR / "metadata.json"


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


try:
    model, metadata = load_artifacts()
except FileNotFoundError as exc:
    st.error(
        f"**Missing file:** `{Path(exc.filename).name}`. Keep all project files together in one "
        "folder and run `python -m streamlit run app.py` from inside it."
    )
    st.stop()
except Exception as exc:  # unpickling problems, usually a library-version mismatch
    st.error(
        f"**Could not load the model** ({type(exc).__name__}: {exc}).\n\n"
        "The model needs scikit-learn 1.6 or newer. Fix with:\n\n`pip install -U -r requirements.txt`"
    )
    st.stop()

BEST_THRESHOLD = metadata["best_threshold"]
CAT_OPTIONS = metadata["categorical_options"]
NUM_RANGES = metadata["numeric_ranges"]

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(hero_html(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Intake form
# ---------------------------------------------------------------------------
with st.form("student_form"):
    st.subheader("Tell us about your current routine")
    tab1, tab2, tab3 = st.tabs(["👤 About you", "📚 Study & stress", "🌿 Health & habits"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", CAT_OPTIONS["Gender"])
            age = st.slider("Age", int(NUM_RANGES["Age"][0]), int(NUM_RANGES["Age"][1]), 21)
            profession = st.selectbox("Profession", CAT_OPTIONS["Profession"],
                                      index=CAT_OPTIONS["Profession"].index("Student"))
        with c2:
            degree = st.selectbox("Degree", CAT_OPTIONS["Degree"])
            city = st.text_input("City (optional)", value="Delhi")
            cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            academic_pressure = st.slider("Academic pressure (0 = none, 5 = extreme)", 0.0, 5.0, 3.0, 1.0)
            study_satisfaction = st.slider("Study satisfaction (0 = low, 5 = high)", 0.0, 5.0, 3.0, 1.0)
        with c2:
            work_study_hours = st.slider("Study/work hours per day", 0.0,
                                         float(NUM_RANGES["Work/Study Hours"][1]), 6.0, 0.5)
            financial_stress = st.slider("Financial stress (1 = low, 5 = high)", 1.0, 5.0, 3.0, 1.0)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            sleep_duration = st.selectbox("Sleep duration",
                                          ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], index=2)
            dietary_habits = st.selectbox("Dietary habits", [o for o in CAT_OPTIONS["Dietary Habits"] if o != "Others"])
        with c2:
            family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])
            suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])

    submitted = st.form_submit_button("Check my risk", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if submitted:
    raw_row = {
        "Gender": gender, "Age": age, "City": city, "Profession": profession,
        "Academic Pressure": academic_pressure, "Work Pressure": STUDENT_FIXED_INPUTS["Work Pressure"],
        "CGPA": cgpa, "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": STUDENT_FIXED_INPUTS["Job Satisfaction"],
        "Sleep Duration": sleep_duration, "Dietary Habits": dietary_habits, "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours, "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
    }
    input_df = add_domain_features(raw_row)[metadata["feature_columns"]]
    prob = float(model.predict_proba(input_df)[:, 1][0])
    tier = risk_tier(prob, BEST_THRESHOLD)
    recs = recommendation_engine(raw_row, prob, BEST_THRESHOLD)

    st.markdown("## Your results")
    if suicidal_thoughts == "Yes":  # safety first: shown above everything else
        st.error(CRISIS_MESSAGE)

    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.markdown(f'<div class="card">{gauge_svg(prob, BEST_THRESHOLD)}'
                    f'<div class="small" style="text-align:center">Estimated risk score</div></div>',
                    unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="card">{tier_html(tier)}'
                    f'<div class="small" style="margin-top:.7rem">Alert threshold: {BEST_THRESHOLD * 100:.0f}% '
                    "(tuned to catch most at-risk students early, so some false alarms are expected).</div></div>",
                    unsafe_allow_html=True)

    # --- What is driving the estimate ---
    try:
        factors = explain_factors(model, input_df)
        up = [(describe_factor(k, raw_row), v) for k, v in factors if v > 0.05][:4]
        down = [(describe_factor(k, raw_row), v) for k, v in reversed(factors) if v < -0.05][:3]
        st.markdown("### What is influencing this estimate")
        f1, f2 = st.columns(2, gap="medium")
        f1.markdown(f'<div class="card"><b>Pushing the estimate up</b>{factor_rows_html(up, "#dc2626")}</div>',
                    unsafe_allow_html=True)
        f2.markdown(f'<div class="card"><b>Pulling it down</b>{factor_rows_html(down, "#16a34a")}</div>',
                    unsafe_allow_html=True)
        st.caption("Relative influence of each group of answers inside the model. These are statistical "
                   "patterns in the survey data, not causes.")
    except Exception:  # explanation is a nice-to-have; never block the result
        factors = []

    # --- Recommendations ---
    st.markdown("### Suggested next steps")
    icons = {"Immediate Support": "🚨", "Academic Strategies": "📚",
             "Wellness & Lifestyle": "🌿", "Professional Guidance": "💼"}
    shown = {c: [r for r in items if r != CRISIS_MESSAGE] for c, items in recs.items()}
    cards = [(c, items) for c, items in shown.items() if items]
    for i in range(0, len(cards), 2):
        cols = st.columns(2, gap="medium")
        for col, (cat, items) in zip(cols, cards[i:i + 2]):
            with col, st.container(border=True):
                st.markdown(f"**{icons[cat]} {cat}**")
                for r in items:
                    st.markdown(f"- {r}")

    summary = "\n".join(
        [f"Burnout risk check: {prob * 100:.0f}% ({tier} risk)", f"Alert threshold: {BEST_THRESHOLD * 100:.0f}%", ""]
        + [f"{c}:\n" + "\n".join(f"  - {r.replace('**', '')}" for r in items)
           for c, items in recs.items() if items]
        + ["", "Research prototype, not a diagnosis."]
    )
    st.download_button("⬇️ Download summary (to share with a counselor)", summary,
                       file_name="burnout_check_summary.txt", use_container_width=True)

    with st.expander("Model details & transparency"):
        st.write(f"**Model:** {metadata['best_model_name']} (Platt-calibrated)")
        st.write(f"**Training target:** `{metadata['target']}` label from a student survey, used here as a "
                 "proxy for burnout risk (the dataset has no direct burnout label).")
        st.write(f"**Decision threshold:** {BEST_THRESHOLD:.2f}")
        st.write("**Held-out test performance:**")
        st.json(metadata["final_test_metrics"])

st.divider()
st.caption(
    "⚠️ Research/decision-support prototype trained on a public student-depression survey dataset. "
    "Not a substitute for professional mental-health assessment. If you or someone you know is in "
    "crisis, contact a counsellor or local helpline immediately."
)
