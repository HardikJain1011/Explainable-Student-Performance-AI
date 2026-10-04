"""Synthetic student-performance dataset.

The target is generated from a known (noisy) formula, so the explanations
the models produce can be sanity-checked against ground truth:
attendance, previous GPA, exam scores, assignments and study hours push
performance up; extreme sleep deviation and too many extracurriculars push it down.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "attendance",             # % of classes attended
    "prev_gpa",               # previous semester GPA (0-10 scale)
    "study_hours",            # average hours/day
    "assignments_completed",  # % of assignments submitted
    "sleep_hours",            # average hours/night
    "extracurriculars",       # number of activities/clubs
    "prev_exam_score",        # previous exam score (%)
]
CLASS_NAMES = ["Low", "Medium", "High"]

FEATURE_LABELS = {
    "attendance": "Attendance (%)",
    "prev_gpa": "Previous GPA",
    "study_hours": "Study hours / day",
    "assignments_completed": "Assignments completed (%)",
    "sleep_hours": "Sleep hours / night",
    "extracurriculars": "Extracurricular activities",
    "prev_exam_score": "Previous exam score (%)",
}


def generate_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(78, 15, n), 30, 100)
    prev_gpa = np.clip(rng.normal(7.2, 1.2, n), 4, 10)
    study_hours = np.clip(rng.normal(3.5, 1.5, n), 0, 10)
    assignments = np.clip(rng.normal(80, 18, n), 20, 100)
    sleep = np.clip(rng.normal(6.8, 1.2, n), 3, 10)
    extra = np.clip(rng.poisson(1.5, n), 0, 6)
    # exam score correlates with GPA but is not identical to it
    exam = np.clip(prev_gpa * 8.5 + rng.normal(8, 8, n), 20, 100)

    score = (
        0.035 * (attendance - 75)
        + 0.90 * (prev_gpa - 7)
        + 0.25 * (study_hours - 3.5)
        + 0.020 * (assignments - 75)
        - 0.30 * np.abs(sleep - 7.5)
        + 0.030 * (exam - 70)
        + 0.10 * np.minimum(extra, 3)
        - 0.15 * np.maximum(extra - 3, 0)
        + rng.normal(0, 0.6, n)
    )
    lo, hi = np.quantile(score, [0.30, 0.70])
    target = np.where(score < lo, 0, np.where(score < hi, 1, 2))

    df = pd.DataFrame({
        "attendance": attendance.round(1),
        "prev_gpa": prev_gpa.round(2),
        "study_hours": study_hours.round(1),
        "assignments_completed": assignments.round(1),
        "sleep_hours": sleep.round(1),
        "extracurriculars": extra,
        "prev_exam_score": exam.round(1),
        "performance": target,
    })
    return df
