"""
Explainability module for FairSight.

Implements a lightweight SHAP-style feature attribution using
marginal contribution estimation (no C extensions required).

For each applicant, we estimate how much each feature pushes the
model score above or below the population mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature display names and their direction of bias
FEATURE_META = {
    "technical_score": {
        "label": "Technical Skill Score",
        "direction": "positive",
        "bias_related": False,
    },
    "interview_score": {
        "label": "Interview Score",
        "direction": "positive",
        "bias_related": False,
    },
    "years_experience": {
        "label": "Years of Experience",
        "direction": "positive",
        "bias_related": False,
    },
    "gender_coded_keywords": {
        "label": "Gender-Coded Keywords in Resume",
        "direction": "negative",
        "bias_related": True,
    },
    "womens_college": {
        "label": "Attended All-Women's College",
        "direction": "negative",
        "bias_related": True,
    },
}

# Approximate model weights (mirrors generate.py scoring logic)
_WEIGHTS = {
    "technical_score": 0.40 / 100,
    "interview_score": 0.35 / 100,
    "years_experience": 0.25 / 15,
    "gender_coded_keywords": -0.04,
    "womens_college": -0.10,
}


def _predicted_score(row: pd.Series) -> float:
    score = sum(row[feat] * w for feat, w in _WEIGHTS.items())
    return float(np.clip(score, 0, 1))


def global_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global feature importance as the absolute weighted contribution
    of each feature across all applicants.
    """
    records = []
    for feat, weight in _WEIGHTS.items():
        mean_val = df[feat].mean()
        # Importance = |weight| * std(feature) — measures spread of impact
        std_val = df[feat].std()
        importance = abs(weight) * std_val
        records.append(
            {
                "feature": FEATURE_META[feat]["label"],
                "importance": round(importance, 4),
                "bias_related": FEATURE_META[feat]["bias_related"],
                "direction": FEATURE_META[feat]["direction"],
                "mean_value": round(mean_val, 2),
            }
        )
    result = pd.DataFrame(records).sort_values("importance", ascending=False)
    return result


def individual_shap(row: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate SHAP-style attributions for a single applicant.

    Attribution for feature f = weight_f * (applicant_value_f - mean_f)
    This is the linear SHAP formula for additive models.
    """
    records = []
    for feat, weight in _WEIGHTS.items():
        mean_val = df[feat].mean()
        attribution = weight * (row[feat] - mean_val)
        records.append(
            {
                "feature": FEATURE_META[feat]["label"],
                "attribution": round(attribution, 4),
                "applicant_value": row[feat],
                "population_mean": round(mean_val, 2),
                "bias_related": FEATURE_META[feat]["bias_related"],
            }
        )
    result = pd.DataFrame(records).sort_values("attribution", key=abs, ascending=False)
    return result


def plain_english_explanation(row: pd.Series, df: pd.DataFrame, top_n: int = 3) -> list[str]:
    """
    Generate plain-English sentences explaining the top factors.
    """
    shap_df = individual_shap(row, df)
    explanations = []

    for _, r in shap_df.head(top_n).iterrows():
        feat_label = r["feature"]
        attr = r["attribution"]
        val = r["applicant_value"]
        mean = r["population_mean"]

        if attr > 0:
            direction_word = "boosted"
            comparison = f"({val:.1f} vs average {mean:.1f})" if not r["bias_related"] else ""
        else:
            direction_word = "reduced"
            comparison = f"({val:.1f} vs average {mean:.1f})" if not r["bias_related"] else ""

        if r["bias_related"] and feat_label == "Gender-Coded Keywords in Resume":
            if val > 0:
                explanations.append(
                    f"⚠️ The presence of {int(val)} gender-coded keyword(s) in the resume "
                    f"reduced the score — this is a known bias factor from the Amazon 2018 case study."
                )
            else:
                explanations.append(
                    "✅ No gender-coded keywords were detected in the resume."
                )
        elif r["bias_related"] and feat_label == "Attended All-Women's College":
            if val == 1:
                explanations.append(
                    "⚠️ Attendance at an all-women's college reduced the score — "
                    "this mirrors the discriminatory pattern identified in Amazon's 2018 AI tool."
                )
            else:
                explanations.append(
                    "✅ University type did not negatively affect the score."
                )
        else:
            explanations.append(
                f"{'📈' if attr > 0 else '📉'} {feat_label} {direction_word} the score {comparison}."
            )

    return explanations
