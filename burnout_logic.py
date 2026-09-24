"""
Pure (Streamlit-free) logic for the burnout risk app: feature engineering,
risk tiers and the recommendation engine. Kept separate from app.py so it can
be unit-tested without a running Streamlit session.
"""
import numpy as np
import pandas as pd

# Probability at/above which a result is always "High". The lower tier boundary
# is the model's tuned decision threshold (metadata["best_threshold"]), so the
# tier and the early-warning alert can never contradict each other.
HIGH_CUTOFF = 0.65

# Work Pressure / Job Satisfaction were ~constant 0 in the training data
# (std ~0.04), so any non-zero value is a wild extrapolation for the model.
# They are fixed to 0 instead of being exposed as inputs.
STUDENT_FIXED_INPUTS = {"Work Pressure": 0.0, "Job Satisfaction": 0.0}

# Verified against Govt. of India sources (PIB / MoHFW): Tele-MANAS, 24x7, toll-free.
CRISIS_MESSAGE = (
    "🚨 **CRITICAL:** You mentioned suicidal thoughts. Please talk to someone right now. "
    "In India, **Tele-MANAS** is free and open 24/7: call **14416** or **1-800-891-4416**. "
    "If you are in immediate danger, call your local emergency number (112 in India). "
    "Outside India, find a local line at https://findahelpline.com. You are not alone, and help is available."
)


def normalize_sleep_hours(value: str):
    mapping = {
        "less than 5 hours": 4.0, "5-6 hours": 5.5, "7-8 hours": 7.5,
        "more than 8 hours": 9.0, "5-6": 5.5, "7-8": 7.5, "8-9": 8.5,
        "10-11": 10.5, "2-3 hours": 2.5,
    }
    return mapping.get(str(value).strip().lower(), np.nan)


def add_domain_features(row: dict) -> pd.DataFrame:
    """Must mirror the training notebook's feature engineering exactly."""
    out = dict(row)
    ap = float(out.get("Academic Pressure", np.nan))
    ss = float(out.get("Study Satisfaction", np.nan))
    wsh = float(out.get("Work/Study Hours", np.nan))
    fs = float(out.get("Financial Stress", np.nan))

    out["Academic_Strain"] = ap - ss
    out["Low_Study_Satisfaction"] = float(ss <= 3)  # 3 ~ dataset median
    out["Workload_Burden"] = wsh * (1 + ap)
    out["Financial_Academic_Strain"] = fs * (1 + ap)
    sleep_hours = normalize_sleep_hours(out.get("Sleep Duration", ""))
    out["Sleep_Hours_Approx"] = sleep_hours
    out["Sleep_Work_Imbalance"] = (
        wsh / sleep_hours if sleep_hours and sleep_hours != 0 else np.nan
    )
    return pd.DataFrame([out])


def risk_tier(prob: float, threshold: float) -> str:
    """Low = below the early-warning threshold; High = at/above HIGH_CUTOFF."""
    if prob < threshold:
        return "Low"
    if prob < max(HIGH_CUTOFF, threshold):
        return "Medium"
    return "High"


def recommendation_engine(row: dict, probability: float, threshold: float):
    recs = {
        "Immediate Support": [],
        "Academic Strategies": [],
        "Wellness & Lifestyle": [],
        "Professional Guidance": [],
    }
    tier = risk_tier(probability, threshold)

    def num(col):
        try:
            return float(row.get(col, np.nan))
        except Exception:
            return np.nan

    academic_pressure = num("Academic Pressure")
    study_satisfaction = num("Study Satisfaction")
    work_hours = num("Work/Study Hours")
    financial_stress = num("Financial Stress")
    cgpa = num("CGPA")
    suicidal_thoughts = row.get("Have you ever had suicidal thoughts ?", "No")
    sleep_val = normalize_sleep_hours(row.get("Sleep Duration", ""))

    # --- 1. Immediate Support (crisis alert fires regardless of probability) ---
    if suicidal_thoughts == "Yes":
        recs["Immediate Support"].append(CRISIS_MESSAGE)

    if tier == "High":
        recs["Immediate Support"].append(
            "Early-warning alert: High risk detected. We strongly recommend reaching "
            "out to a qualified mental health counselor."
        )
    elif tier == "Medium":
        recs["Immediate Support"].append(
            "Moderate risk: Consider scheduling a check-in with a counselor to prevent escalation."
        )

    # --- 2. Academic Strategies ---
    if not np.isnan(academic_pressure) and academic_pressure >= 4:
        recs["Academic Strategies"].append(
            "Review academic workload; try the Pomodoro technique to break large tasks into manageable chunks."
        )
    if not np.isnan(study_satisfaction) and study_satisfaction <= 2:
        recs["Academic Strategies"].append(
            "Connect with a peer study group or an academic mentor to find new ways to engage with your subjects."
        )
    if not np.isnan(cgpa) and cgpa < 6:
        recs["Academic Strategies"].append(
            "Explore campus tutoring services or faculty office hours for targeted academic support."
        )

    # --- 3. Wellness & Lifestyle ---
    if not np.isnan(work_hours) and work_hours >= 8:
        recs["Wellness & Lifestyle"].append(
            "Reduce prolonged study/work sessions. Schedule 'non-negotiable' downtime every day to recharge."
        )

    if not np.isnan(work_hours) and work_hours >= 8 and (not np.isnan(sleep_val) and sleep_val < 6):
        recs["Wellness & Lifestyle"].append(
            "⚠️ **Burnout Cycle Alert:** You are working long hours with very little sleep. "
            "This is a high-strain pattern; prioritize sleep immediately to avoid total exhaustion."
        )
    elif not np.isnan(sleep_val) and sleep_val < 6:
        recs["Wellness & Lifestyle"].append(
            "Prioritize a consistent sleep-wake cycle. Aim for 7-9 hours to improve cognitive function and mood."
        )

    if row.get("Dietary Habits") == "Unhealthy":
        recs["Wellness & Lifestyle"].append(
            "Regular, balanced meals support energy and mood. Try small changes first, "
            "such as not skipping breakfast or adding one home-cooked meal a day."
        )

    # --- 4. Professional Guidance ---
    if not np.isnan(financial_stress) and financial_stress >= 4:
        recs["Professional Guidance"].append(
            "Consider connecting with student financial aid offices or scholarship advisors to reduce financial anxiety."
        )

    if not any(recs.values()):
        recs["Wellness & Lifestyle"].append(
            "Continue maintaining your healthy study, sleep, and social-support habits!"
        )

    return recs


# ---------------------------------------------------------------------------
# Explainability: per-factor contribution to the model's log-odds
# ---------------------------------------------------------------------------
FACTOR_GROUPS = {
    "Suicidal thoughts": ["Have you ever had suicidal thoughts ?"],
    "Academic pressure & satisfaction": ["Academic Pressure", "Study Satisfaction", "Academic_Strain", "Low_Study_Satisfaction"],
    "Workload & sleep": ["Work/Study Hours", "Workload_Burden", "Sleep Duration", "Sleep_Hours_Approx", "Sleep_Work_Imbalance"],
    "Financial stress": ["Financial Stress", "Financial_Academic_Strain"],
    "Age": ["Age"],
    "CGPA": ["CGPA"],
    "Dietary habits": ["Dietary Habits"],
    "Family history": ["Family History of Mental Illness"],
    "Background (gender, degree, city, profession)": ["Gender", "Degree", "City", "Profession"],
}
_RAW_TO_GROUP = {raw: g for g, raws in FACTOR_GROUPS.items() for raw in raws}


def explain_factors(model, input_df: pd.DataFrame):
    """Return [(group, log-odds contribution)] sorted high->low.

    Exact for the underlying Logistic Regression: contributions + intercept equal the
    pipeline's decision function. Platt calibration is a monotonic rescale on top of
    it, so the *direction* of each factor carries over to the final probability.
    """
    pipe = model.calibrated_classifiers_[0].estimator.estimator
    pre, clf = pipe.named_steps["preprocessor"], pipe.named_steps["model"]
    x = pre.transform(input_df)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contrib = x[0] * clf.coef_[0]
    raws = sorted(input_df.columns, key=len, reverse=True)  # longest match first
    totals = {}
    for name, c in zip(pre.get_feature_names_out(), contrib):
        rest = name.split("__", 1)[1]
        raw = next((r for r in raws if rest == r or rest.startswith(r + "_")), None)
        group = _RAW_TO_GROUP.get(raw)
        if group:  # Work Pressure / Job Satisfaction are fixed inputs -> baseline, not shown
            totals[group] = totals.get(group, 0.0) + float(c)
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def describe_factor(group: str, row: dict) -> str:
    """Attach the user's own answer to a factor label, e.g. 'Financial stress: 5/5'."""
    g = lambda k: row.get(k)  # noqa: E731
    detail = {
        "Suicidal thoughts": g("Have you ever had suicidal thoughts ?"),
        "Academic pressure & satisfaction": f"pressure {g('Academic Pressure'):g}/5, satisfaction {g('Study Satisfaction'):g}/5",
        "Workload & sleep": f"{g('Work/Study Hours'):g} h/day, sleep {g('Sleep Duration')}",
        "Financial stress": f"{g('Financial Stress'):g}/5",
        "Age": g("Age"),
        "CGPA": g("CGPA"),
        "Dietary habits": g("Dietary Habits"),
        "Family history": g("Family History of Mental Illness"),
    }.get(group)
    return f"{group}: {detail}" if detail is not None else group
