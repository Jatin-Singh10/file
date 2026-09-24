"""Run with:  pip install pytest && pytest -q"""
import itertools
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from burnout_logic import (
    CRISIS_MESSAGE, add_domain_features, normalize_sleep_hours,
    explain_factors, recommendation_engine, risk_tier,
)
from burnout_ui import gauge_svg, tier_html

HERE = Path(__file__).parent
MODEL = joblib.load(HERE / "burnout_model.joblib")
META = json.loads((HERE / "metadata.json").read_text())
T = META["best_threshold"]

BASE = {
    "Gender": "Female", "Age": 21, "City": "Delhi", "Profession": "Student",
    "Academic Pressure": 3.0, "Work Pressure": 0.0, "CGPA": 7.5,
    "Study Satisfaction": 3.0, "Job Satisfaction": 0.0, "Sleep Duration": "7-8 hours",
    "Dietary Habits": "Moderate", "Degree": "B.Tech",
    "Have you ever had suicidal thoughts ?": "No", "Work/Study Hours": 6.0,
    "Financial Stress": 3.0, "Family History of Mental Illness": "No",
}


APP_DEFAULT_ROW = {**BASE, "Degree": "B.Arch", "Dietary Habits": "Healthy"}  # what the form pre-selects


def prob(row):
    df = add_domain_features(row)[META["feature_columns"]]
    return float(MODEL.predict_proba(df)[:, 1][0])


def test_features_match_metadata_columns():
    df = add_domain_features(BASE)
    assert set(META["feature_columns"]) <= set(df.columns)
    assert 0.0 <= prob(BASE) <= 1.0


def test_zero_hours_and_unknown_sleep_do_not_crash():
    assert add_domain_features({**BASE, "Work/Study Hours": 0.0})["Sleep_Work_Imbalance"][0] == 0.0
    assert np.isnan(normalize_sleep_hours("Others"))
    assert 0 <= prob({**BASE, "Sleep Duration": "Others"}) <= 1


def test_tier_and_alert_never_contradict():
    grid = itertools.product([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5], ["No", "Yes"])
    for ap, ss, fs, st_ in grid:
        p = prob({**BASE, "Academic Pressure": ap, "Study Satisfaction": ss, "Financial Stress": fs,
                  "Have you ever had suicidal thoughts ?": st_})
        assert (risk_tier(p, T) != "Low") == (p >= T)


@pytest.mark.parametrize("p,expected", [(0.0, "Low"), (T - 1e-9, "Low"), (T, "Medium"),
                                        (0.649, "Medium"), (0.65, "High"), (1.0, "High")])
def test_tier_boundaries(p, expected):
    assert risk_tier(p, T) == expected


def test_crisis_message_has_helpline_even_at_low_risk():
    row = {**BASE, "Have you ever had suicidal thoughts ?": "Yes"}
    recs = recommendation_engine(row, 0.01, T)
    assert CRISIS_MESSAGE in recs["Immediate Support"]
    assert "14416" in CRISIS_MESSAGE
    assert CRISIS_MESSAGE not in recommendation_engine(BASE, 0.99, T)["Immediate Support"]


def test_recommendation_rules():
    r = recommendation_engine({**BASE, "Work/Study Hours": 10, "Sleep Duration": "Less than 5 hours",
                               "Dietary Habits": "Unhealthy", "Financial Stress": 5,
                               "Academic Pressure": 5, "Study Satisfaction": 1, "CGPA": 5}, 0.8, T)
    assert any("Burnout Cycle" in s for s in r["Wellness & Lifestyle"])
    assert any("meals" in s for s in r["Wellness & Lifestyle"])
    assert r["Professional Guidance"] and len(r["Academic Strategies"]) == 3
    assert recommendation_engine(BASE, 0.05, T)["Wellness & Lifestyle"]  # fallback


def _run_app():
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    return at


def test_app_smoke_and_ui_consistency():
    at = _run_app()
    html = " ".join(m.value for m in at.markdown)
    tiers = set(re.findall(r'data-tier="(\w+)"', html))
    assert len(tiers) == 1  # badge and banner agree
    assert tiers == {risk_tier(prob(APP_DEFAULT_ROW), T)}
    assert "Pushing the estimate up" in html and "<svg" in html
    assert not at.error  # no crisis card for default inputs


def test_crisis_card_shown_first_when_flagged():
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=60).run()
    next(s for s in at.selectbox if s.label.startswith("Ever had suicidal")).select("Yes")
    at.button[0].click().run()
    assert not at.exception and any("14416" in e.value for e in at.error)


def test_explanation_reconstructs_model_output():
    df = add_domain_features({**BASE, "Academic Pressure": 4.0, "Sleep Duration": "5-6 hours",
                              "Dietary Habits": "Unhealthy"})[META["feature_columns"]]
    pipe = MODEL.calibrated_classifiers_[0].estimator.estimator
    factors = explain_factors(MODEL, df)
    assert factors and factors == sorted(factors, key=lambda kv: kv[1], reverse=True)
    from burnout_logic import _RAW_TO_GROUP  # all groups accounted for except fixed inputs
    total = sum(v for _, v in factors)
    fixed = pipe.decision_function(df)[0] - pipe.named_steps["model"].intercept_[0] - total
    assert abs(fixed) < 0.5  # only Work Pressure / Job Satisfaction (constant baseline) are left out


def test_ui_helpers_render_valid_fragments():
    svg = gauge_svg(0.5, T)
    assert svg.startswith("<svg") and svg.count("<path") == 3 and "50%" in svg
    for extreme in (0.0, 1.0):
        assert "nan" not in gauge_svg(extreme, T).lower()
    assert 'data-tier="High"' in tier_html("High")


def test_degenerate_inputs_not_exposed_and_ranges_match_training():
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=60).run()
    labels = [s.label for s in at.slider]
    assert not any("Work Pressure" in l or "Job Satisfaction" in l for l in labels)
    age = next(s for s in at.slider if s.label == "Age")
    hrs = next(s for s in at.slider if s.label.startswith("Study/work hours"))
    assert age.min >= META["numeric_ranges"]["Age"][0]
    assert hrs.max <= META["numeric_ranges"]["Work/Study Hours"][1]


def test_missing_files_show_friendly_error(tmp_path):
    import shutil
    import streamlit as st
    st.cache_resource.clear()  # cache is process-global; earlier tests already loaded the model
    for f in ("app.py", "burnout_logic.py", "metadata.json"):  # model file deliberately absent
        shutil.copy(HERE / f, tmp_path / f)
    at = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Missing file" in e.value and "burnout_model.joblib" in e.value for e in at.error)


def test_factor_labels_include_user_answer():
    from burnout_logic import describe_factor
    assert describe_factor("Suicidal thoughts", BASE) == "Suicidal thoughts: No"
    assert describe_factor("Financial stress", {**BASE, "Financial Stress": 5.0}) == "Financial stress: 5/5"
    assert "sleep 7-8 hours" in describe_factor("Workload & sleep", BASE)
    assert describe_factor("Background (gender, degree, city, profession)", BASE).startswith("Background")
