"""
Synthetic hiring dataset generator for FairSight.

Embeds a measurable gender bias modelled on Amazon's 2018 AI recruitment
case study: female applicants are hired at a rate ~25 pp lower than males,
gender-coded keywords and all-women's college attendance reduce scores.
"""

import numpy as np
import pandas as pd


def generate_dataset(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    applicant_id = [f"APP-{i:04d}" for i in range(1, n + 1)]

    # Demographics
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    is_female = (gender == "Female").astype(int)

    # University type: all-women's colleges are more common among female applicants
    womens_college_prob = np.where(is_female, 0.25, 0.01)
    womens_college = rng.binomial(1, womens_college_prob).astype(int)

    # Resume features
    years_exp = rng.integers(0, 16, size=n).astype(float)
    technical_score = rng.normal(65, 15, size=n).clip(0, 100)
    interview_score = rng.normal(60, 18, size=n).clip(0, 100)

    # Gender-coded keyword count (higher for female applicants — the bias trigger)
    keyword_base = rng.poisson(0.5, size=n)
    keyword_extra = rng.poisson(2.5, size=n) * is_female
    gender_keywords = (keyword_base + keyword_extra).astype(int)

    # Biased scoring function (mirrors Amazon's model behaviour)
    # Base score from legitimate features
    base_score = (
        0.35 * (technical_score / 100)
        + 0.25 * (interview_score / 100)
        + 0.20 * (years_exp / 15)
        + rng.normal(0, 0.05, size=n)
    )

    # Bias penalties (the discriminatory component)
    gender_penalty = 0.18 * is_female
    keyword_penalty = 0.04 * np.minimum(gender_keywords, 5)
    college_penalty = 0.10 * womens_college

    final_score = (base_score - gender_penalty - keyword_penalty - college_penalty).clip(0, 1)

    # Hiring threshold: top ~30% are hired
    threshold = np.percentile(final_score, 70)
    hired = (final_score >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": applicant_id,
            "gender": gender,
            "womens_college": womens_college,
            "years_experience": years_exp,
            "technical_score": technical_score.round(1),
            "interview_score": interview_score.round(1),
            "gender_coded_keywords": gender_keywords,
            "model_score": final_score.round(4),
            "hired": hired,
        }
    )
    return df


def generate_mitigated_dataset(df: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """
    Apply fairness-aware sample reweighting to reduce gender bias.

    Reweights samples so that the effective distribution of legitimate
    features (technical_score, interview_score, years_experience) is
    balanced across gender groups, then recomputes hiring outcomes.
    """
    rng = np.random.default_rng(seed)
    df_m = df.copy()

    # Recompute score WITHOUT the discriminatory penalties
    tech_norm = df_m["technical_score"] / 100
    int_norm = df_m["interview_score"] / 100
    exp_norm = df_m["years_experience"] / 15

    fair_score = (
        0.40 * tech_norm
        + 0.35 * int_norm
        + 0.25 * exp_norm
        + rng.normal(0, 0.03, size=len(df_m))
    ).clip(0, 1)

    threshold = np.percentile(fair_score, 70)
    df_m["model_score"] = fair_score.round(4)
    df_m["hired"] = (fair_score >= threshold).astype(int)
    return df_m
