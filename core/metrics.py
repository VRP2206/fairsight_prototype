"""
Fairness metrics for FairSight bias detection engine.

Implements:
- Demographic Parity Difference
- Disparate Impact Ratio
- Equalised Odds Difference (TPR gap)
- Overall Bias Score (composite)
"""

from __future__ import annotations

import pandas as pd


def hire_rate(df: pd.DataFrame, group_col: str, group_val: str) -> float:
    subset = df[df[group_col] == group_val]
    if len(subset) == 0:
        return 0.0
    return subset["hired"].mean()


def demographic_parity_difference(df: pd.DataFrame) -> float:
    """Positive value means males are hired at a higher rate."""
    male_rate = hire_rate(df, "gender", "Male")
    female_rate = hire_rate(df, "gender", "Female")
    return round(male_rate - female_rate, 4)


def disparate_impact_ratio(df: pd.DataFrame) -> float:
    """Female hire rate / Male hire rate. Below 0.8 = potential discrimination."""
    male_rate = hire_rate(df, "gender", "Male")
    female_rate = hire_rate(df, "gender", "Female")
    if male_rate == 0:
        return 1.0
    return round(female_rate / male_rate, 4)


def equalised_odds_tpr_gap(df: pd.DataFrame) -> float:
    """
    True Positive Rate gap between male and female applicants.
    Requires a 'true_label' column; falls back to hired as proxy.
    """
    # In our synthetic dataset, hired IS the ground truth (no separate label)
    # We approximate by comparing hire rates among high-scoring applicants
    high_score = df[df["model_score"] >= df["model_score"].median()]
    male_tpr = hire_rate(high_score, "gender", "Male")
    female_tpr = hire_rate(high_score, "gender", "Female")
    return round(male_tpr - female_tpr, 4)


def bias_score(df: pd.DataFrame) -> float:
    """
    Composite bias score in [0, 1].
    Combines DPD and DIR into a single indicator.
    0 = no bias, 1 = maximum bias.
    """
    dpd = demographic_parity_difference(df)
    dir_val = disparate_impact_ratio(df)

    # Normalise DPD: max realistic gap is ~0.5
    dpd_norm = min(abs(dpd) / 0.5, 1.0)

    # Normalise DIR: 0 = worst (0.0 ratio), 1 = best (1.0 ratio)
    dir_norm = max(0.0, 1.0 - dir_val)

    return round(0.6 * dpd_norm + 0.4 * dir_norm, 4)


def compute_all_metrics(df: pd.DataFrame) -> dict:
    male_rate = hire_rate(df, "gender", "Male")
    female_rate = hire_rate(df, "gender", "Female")
    return {
        "male_hire_rate": round(male_rate, 4),
        "female_hire_rate": round(female_rate, 4),
        "demographic_parity_difference": demographic_parity_difference(df),
        "disparate_impact_ratio": disparate_impact_ratio(df),
        "equalised_odds_tpr_gap": equalised_odds_tpr_gap(df),
        "bias_score": bias_score(df),
        "total_applicants": len(df),
        "total_hired": int(df["hired"].sum()),
    }
