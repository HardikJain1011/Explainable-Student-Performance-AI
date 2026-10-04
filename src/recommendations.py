"""Actionable, model-based improvement suggestions and what-if simulations."""

from itertools import product

import numpy as np
import pandas as pd

from .data import FEATURES, FEATURE_LABELS

# Only include factors a student can reasonably change in the short term.
ACTIONABLE = [
    "attendance",
    "study_hours",
    "assignments_completed",
    "sleep_hours",
    "extracurriculars",
]

RANGES = {
    "attendance": (30.0, 100.0),
    "study_hours": (0.0, 10.0),
    "assignments_completed": (20.0, 100.0),
    "sleep_hours": (3.0, 10.0),
    "extracurriculars": (0.0, 6.0),
}

STEP = {
    "attendance": 5.0,
    "study_hours": 0.5,
    "assignments_completed": 5.0,
    "sleep_hours": 0.5,
    "extracurriculars": 1.0,
}

# Relative effort weights: lower means the simulator prefers this change.
EFFORT = {
    "attendance": 1.0,
    "study_hours": 1.0,
    "assignments_completed": 0.8,
    "sleep_hours": 0.7,
    "extracurriculars": 0.5,
}

DESCRIPTIONS = {
    "attendance": "Aim for more consistent class attendance.",
    "study_hours": "Build a steady study routine rather than relying on last-minute study.",
    "assignments_completed": "Complete and submit more assignments on time.",
    "sleep_hours": "Move toward a more regular sleep schedule around 7–8 hours.",
    "extracurriculars": "Keep activities balanced so they do not crowd out academic time.",
}


def _clip_value(feature, value):
    lo, hi = RANGES[feature]
    return float(np.clip(value, lo, hi))


def _fmt_change(feature, old, new):
    labels = {
        "attendance": "attendance",
        "study_hours": "study hours/day",
        "assignments_completed": "assignment completion",
        "sleep_hours": "sleep hours/night",
        "extracurriculars": "extracurricular activities",
    }
    if feature == "extracurriculars":
        return f"{labels[feature]}: {old:.0f} → {new:.0f}"
    if feature in {"attendance", "assignments_completed"}:
        return f"{labels[feature]}: {old:.0f}% → {new:.0f}%"
    return f"{labels[feature]}: {old:.1f} → {new:.1f}"


def candidate_grid(x_row: pd.Series, feature):
    """Generate plausible values around the student's current value."""
    current = float(x_row[feature])
    lo, hi = RANGES[feature]
    step = STEP[feature]
    values = {current}
    # Search both directions, but bias toward changes that could improve the profile.
    for k in range(1, 5):
        values.add(_clip_value(feature, current + k * step))
        values.add(_clip_value(feature, current - k * step))
    if feature == "sleep_hours":
        # Sleep has a non-linear relationship; include the healthy target band.
        values.update({_clip_value(feature, 7.0), _clip_value(feature, 7.5), _clip_value(feature, 8.0)})
    return sorted(values)


def _effort(x_row, candidate):
    score = 0.0
    for feature in ACTIONABLE:
        old = float(x_row[feature])
        new = float(candidate[feature])
        if feature == "extracurriculars":
            denom = 6.0
        elif feature == "study_hours":
            denom = 10.0
        else:
            denom = 100.0
        score += abs(new - old) / denom * EFFORT[feature]
    return score


def find_counterfactuals(model, x_row: pd.Series, desired_class: int, max_changes: int = 3,
                         max_candidates: int = 5):
    """Find small, plausible input changes that move the model to desired_class.

    The search tests combinations of actionable target values rather than inventing
    arbitrary values. This is a model what-if simulation, not a causal intervention.
    """
    base = x_row.copy().astype(float)
    base_df = base.to_frame().T
    base_pred = int(model.predict(base_df)[0])
    if base_pred == desired_class:
        return []

    target_values = {
        "attendance": 95.0,
        "study_hours": 5.0,
        "assignments_completed": 95.0,
        "sleep_hours": 7.5,
        "extracurriculars": 2.0,
    }
    # Prefer realistic improvements: don't recommend moving a value in a direction
    # that is unlikely to help the synthetic ground-truth relationship.
    useful = []
    for feature in ACTIONABLE:
        old = float(base[feature])
        new = target_values[feature]
        if abs(new - old) > 1e-9:
            useful.append(feature)

    results = []
    for n_changes in range(1, min(max_changes, len(useful)) + 1):
        for combo in __import__("itertools").combinations(useful, n_changes):
            candidate = base.copy()
            for feature in combo:
                candidate[feature] = target_values[feature]
            pred = int(model.predict(candidate.to_frame().T)[0])
            if pred != desired_class:
                continue
            proba = model.predict_proba(candidate.to_frame().T)[0]
            results.append({
                "candidate": candidate,
                "changes": list(combo),
                "effort": _effort(base, candidate),
                "confidence": float(proba[desired_class]),
            })
        if results:
            break

    results.sort(key=lambda r: (len(r["changes"]), r["effort"], -r["confidence"]))
    return results[:max_candidates]


def personalized_suggestions(model, x_row: pd.Series, shap_table: pd.DataFrame, pred_idx: int):
    """Create readable suggestions grounded in current values and SHAP direction."""
    suggestions = []
    table = shap_table.copy()
    for _, row in table.iterrows():
        feature = row["feature"]
        if feature not in ACTIONABLE:
            continue
        value = float(row["value"])
        impact = float(row["shap"])
        if impact >= 0:
            continue

        if feature == "attendance" and value < 90:
            suggestions.append((feature, "Raise attendance", f"{value:.0f}% → aim for 90%+", DESCRIPTIONS[feature]))
        elif feature == "study_hours" and value < 5:
            suggestions.append((feature, "Increase focused study", f"{value:.1f} → aim for about 4–5 hours/day", DESCRIPTIONS[feature]))
        elif feature == "assignments_completed" and value < 90:
            suggestions.append((feature, "Complete more assignments", f"{value:.0f}% → aim for 90%+", DESCRIPTIONS[feature]))
        elif feature == "sleep_hours" and abs(value - 7.5) > 1.0:
            suggestions.append((feature, "Regularize sleep", f"{value:.1f} → move toward 7–8 hours/night", DESCRIPTIONS[feature]))
        elif feature == "extracurriculars" and value > 3:
            suggestions.append((feature, "Balance activities", f"{value:.0f} → consider 2–3 activities", DESCRIPTIONS[feature]))

    return suggestions[:4]
