"""
FairSight - AI Auditing and Governance Platform
Streamlit prototype demonstrating bias detection, explainability,
mitigation, and applicant transparency for automated hiring systems.

Case study context: Amazon 2018 biased AI recruitment tool.
"""

import io
import datetime
import sys
import os

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from data.generate import generate_dataset, generate_mitigated_dataset
from core.metrics import compute_all_metrics, disparate_impact_ratio, bias_score
from core.explainability import (
    global_feature_importance,
    individual_shap,
    plain_english_explanation,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FairSight | AI Bias Auditing Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .status-green  { color: #4ade80; font-weight: 700; }
    .status-amber  { color: #fbbf24; font-weight: 700; }
    .status-red    { color: #f87171; font-weight: 700; }
    .section-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
    .case-study-box {
        background: #2d1b4e;
        border-left: 4px solid #a78bfa;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "df_mitigated" not in st.session_state:
    st.session_state.df_mitigated = None
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/scales.png",
        width=64,
    )
    st.title("FairSight ⚖️")
    st.caption("AI Auditing & Governance Platform")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📊 Overview & Bias Metrics",
            "🔍 Explainability",
            "🛠️ Bias Mitigation",
            "👤 Applicant View",
            "📋 Audit Report",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Dataset Controls**")
    n_records = st.slider("Number of applicants", 500, 3000, 1200, 100)
    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("🔄 Generate Dataset", use_container_width=True):
        with st.spinner("Generating synthetic dataset..."):
            st.session_state.df = generate_dataset(n=n_records, seed=int(seed))
            st.session_state.df_mitigated = None
        st.success(f"✅ Generated {n_records} applicant records.")

    st.divider()
    st.caption("Prototype 0.1 · FIT1055 · 2025")

# ---------------------------------------------------------------------------
# Auto-load dataset on first run
# ---------------------------------------------------------------------------
if st.session_state.df is None:
    st.session_state.df = generate_dataset(n=1200, seed=42)

df = st.session_state.df

# ===========================================================================
# PAGE: Overview & Bias Metrics
# ===========================================================================
if page == "📊 Overview & Bias Metrics":
    st.title("📊 Overview & Bias Metrics")
    st.markdown(
        "FairSight continuously audits hiring models for algorithmic bias. "
        "This dashboard surfaces fairness metrics so recruiters and compliance "
        "teams can identify and act on discriminatory patterns before they harm applicants."
    )

    st.markdown(
        """
        <div class="case-study-box">
        <strong>🏢 Case Study Context — Amazon 2018</strong><br>
        Amazon's AI recruitment tool was found to systematically downgrade resumes
        containing the word <em>"women's"</em> and penalise graduates of all-women's colleges.
        The model had learned from 10 years of male-dominated hiring data.
        FairSight is designed to detect and prevent exactly this kind of bias.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = compute_all_metrics(df)
    dir_val = metrics["disparate_impact_ratio"]
    bs = metrics["bias_score"]
    dpd = metrics["demographic_parity_difference"]

    # Status helpers
    def dir_status(v):
        if v >= 0.8:
            return "green", "PASS"
        elif v >= 0.6:
            return "amber", "BORDERLINE"
        return "red", "FAIL"

    def bs_status(v):
        if v < 0.3:
            return "green", "LOW"
        elif v < 0.5:
            return "amber", "MODERATE"
        return "red", "HIGH"

    dir_colour, dir_label = dir_status(dir_val)
    bs_colour, bs_label = bs_status(bs)

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", f"{metrics['total_applicants']:,}")
    col2.metric("Total Hired", f"{metrics['total_hired']:,}")
    col3.metric(
        "Male Hire Rate",
        f"{metrics['male_hire_rate']:.1%}",
    )
    col4.metric(
        "Female Hire Rate",
        f"{metrics['female_hire_rate']:.1%}",
        delta=f"{metrics['female_hire_rate'] - metrics['male_hire_rate']:.1%}",
        delta_color="inverse",
    )

    st.divider()

    # Fairness metrics panel
    st.subheader("Fairness Metrics Panel")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("**Disparate Impact Ratio**")
        st.markdown(
            f"<span style='font-size:2rem; font-weight:700'>{dir_val:.3f}</span> "
            f"<span class='status-{dir_colour}'>● {dir_label}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Female hire rate ÷ Male hire rate. Threshold: ≥ 0.80 (four-fifths rule)")
        if dir_val < 0.8:
            st.warning("⚠️ Potential discriminatory impact detected. DIR is below the 0.80 threshold.")

    with m2:
        st.markdown("**Demographic Parity Difference**")
        st.markdown(
            f"<span style='font-size:2rem; font-weight:700'>{dpd:.3f}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Male hire rate minus Female hire rate. Closer to 0 is fairer.")

    with m3:
        st.markdown("**Overall Bias Score**")
        st.markdown(
            f"<span style='font-size:2rem; font-weight:700'>{bs:.3f}</span> "
            f"<span class='status-{bs_colour}'>● {bs_label}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Composite score 0–1. 0 = no bias, 1 = maximum bias.")
        if bs > 0.5:
            st.error("🚨 High bias detected. Recommend applying mitigation (see Bias Mitigation tab).")

    st.divider()

    # Hire rate comparison chart
    st.subheader("Hire Rate by Gender")
    hire_data = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Hire Rate": [metrics["male_hire_rate"], metrics["female_hire_rate"]],
            "Colour": ["#60a5fa", "#f472b6"],
        }
    )
    bar = (
        alt.Chart(hire_data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Gender:N", axis=alt.Axis(labelFontSize=14)),
            y=alt.Y(
                "Hire Rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", labelFontSize=12),
            ),
            color=alt.Color("Colour:N", scale=None),
            tooltip=[
                alt.Tooltip("Gender:N"),
                alt.Tooltip("Hire Rate:Q", format=".1%"),
            ],
        )
        .properties(height=300)
    )
    threshold_line = (
        alt.Chart(pd.DataFrame({"y": [metrics["male_hire_rate"] * 0.8]}))
        .mark_rule(color="#f87171", strokeDash=[6, 3], strokeWidth=2)
        .encode(y="y:Q")
    )
    st.altair_chart(bar + threshold_line, use_container_width=True)
    st.caption("Red dashed line = 80% of male hire rate (four-fifths rule threshold for female group)")

    # Distribution of model scores by gender
    st.subheader("Model Score Distribution by Gender")
    score_data = df[["gender", "model_score"]].copy()
    hist = (
        alt.Chart(score_data)
        .mark_bar(opacity=0.7, binSpacing=1)
        .encode(
            x=alt.X("model_score:Q", bin=alt.Bin(maxbins=40), title="Model Score"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color(
                "gender:N",
                scale=alt.Scale(domain=["Male", "Female"], range=["#60a5fa", "#f472b6"]),
                legend=alt.Legend(title="Gender"),
            ),
            tooltip=["gender:N", "count():Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(hist, use_container_width=True)

    # Gender-coded keywords impact
    st.subheader("Impact of Gender-Coded Keywords")
    kw_data = (
        df.groupby("gender_coded_keywords")["hired"]
        .mean()
        .reset_index()
        .rename(columns={"hired": "hire_rate"})
    )
    kw_data = kw_data[kw_data["gender_coded_keywords"] <= 8]
    kw_chart = (
        alt.Chart(kw_data)
        .mark_line(point=True, color="#a78bfa", strokeWidth=2)
        .encode(
            x=alt.X("gender_coded_keywords:O", title="Gender-Coded Keyword Count"),
            y=alt.Y("hire_rate:Q", axis=alt.Axis(format=".0%"), title="Hire Rate"),
            tooltip=[
                alt.Tooltip("gender_coded_keywords:O", title="Keywords"),
                alt.Tooltip("hire_rate:Q", format=".1%", title="Hire Rate"),
            ],
        )
        .properties(height=250)
    )
    st.altair_chart(kw_chart, use_container_width=True)
    st.caption(
        "As gender-coded keyword count increases, hire rate drops — "
        "directly replicating the Amazon 2018 bias pattern."
    )


# ===========================================================================
# PAGE: Explainability
# ===========================================================================
elif page == "🔍 Explainability":
    st.title("🔍 Explainability & Transparency")
    st.markdown(
        "Understand how the hiring model makes decisions — both at a global level "
        "across all applicants, and for individual candidates."
    )

    tab1, tab2 = st.tabs(["🌐 Global Feature Importance", "👤 Individual Explanation"])

    with tab1:
        st.subheader("Global Feature Importance")
        st.markdown(
            "The chart below shows how much each feature influences hiring decisions "
            "across the entire dataset. Bias-related features are highlighted in red."
        )
        fi = global_feature_importance(df)
        fi["colour"] = fi["bias_related"].map({True: "#f87171", False: "#60a5fa"})

        fi_chart = (
            alt.Chart(fi)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13)),
                x=alt.X("importance:Q", title="Feature Importance (|weight| × std)"),
                color=alt.Color("colour:N", scale=None),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("importance:Q", format=".4f", title="Importance"),
                    alt.Tooltip("mean_value:Q", format=".2f", title="Population Mean"),
                    alt.Tooltip("bias_related:N", title="Bias-Related"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(fi_chart, use_container_width=True)
        st.markdown(
            "🔴 **Red bars** = bias-related features (gender-coded keywords, all-women's college). "
            "🔵 **Blue bars** = legitimate merit-based features."
        )

        st.markdown(
            """
            <div class="case-study-box">
            <strong>⚠️ Gender Bias Spotlight</strong><br>
            In Amazon's 2018 tool, the model assigned negative weights to resumes mentioning
            "women's" (e.g., "women's chess club") and penalised graduates of all-women's colleges.
            The chart above shows these same features carrying negative weight in our synthetic model.
            A fair model should show near-zero importance for these features.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.subheader("Individual Applicant Explanation")
        applicant_ids = df["applicant_id"].tolist()
        selected_id = st.selectbox("Select Applicant ID", applicant_ids, index=0)
        row = df[df["applicant_id"] == selected_id].iloc[0]

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("**Applicant Profile**")
            outcome_colour = "#4ade80" if row["hired"] == 1 else "#f87171"
            outcome_text = "✅ HIRED" if row["hired"] == 1 else "❌ NOT HIRED"
            st.markdown(
                f"<span style='font-size:1.4rem; color:{outcome_colour}; font-weight:700'>"
                f"{outcome_text}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Model Score:** `{row['model_score']:.4f}`")
            st.markdown(f"**Gender:** {row['gender']}")
            st.markdown(f"**University Type:** {'All-Women\'s College' if row['womens_college'] else 'Mixed/Other'}")
            st.markdown(f"**Years Experience:** {int(row['years_experience'])}")
            st.markdown(f"**Technical Score:** {row['technical_score']:.1f}/100")
            st.markdown(f"**Interview Score:** {row['interview_score']:.1f}/100")
            st.markdown(f"**Gender-Coded Keywords:** {int(row['gender_coded_keywords'])}")

            if row["gender_coded_keywords"] > 0:
                st.warning(
                    f"⚠️ This resume contains {int(row['gender_coded_keywords'])} "
                    "gender-coded keyword(s). This may have negatively affected the score."
                )

        with col_b:
            st.markdown("**SHAP-Style Feature Attributions**")
            shap_df = individual_shap(row, df)
            shap_df["colour"] = shap_df["attribution"].apply(
                lambda v: "#4ade80" if v >= 0 else "#f87171"
            )
            shap_chart = (
                alt.Chart(shap_df)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                .encode(
                    y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=12)),
                    x=alt.X("attribution:Q", title="Attribution (vs population mean)"),
                    color=alt.Color("colour:N", scale=None),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Feature"),
                        alt.Tooltip("attribution:Q", format=".4f", title="Attribution"),
                        alt.Tooltip("applicant_value:Q", format=".2f", title="Applicant Value"),
                        alt.Tooltip("population_mean:Q", format=".2f", title="Population Mean"),
                    ],
                )
                .properties(height=260)
            )
            zero_line = (
                alt.Chart(pd.DataFrame({"x": [0]}))
                .mark_rule(color="white", strokeWidth=1, opacity=0.4)
                .encode(x="x:Q")
            )
            st.altair_chart(shap_chart + zero_line, use_container_width=True)
            st.caption("🟢 Green = pushed score up  |  🔴 Red = pushed score down")

        st.markdown("**Plain-English Explanation**")
        explanations = plain_english_explanation(row, df, top_n=3)
        for exp in explanations:
            st.markdown(f"- {exp}")


# ===========================================================================
# PAGE: Bias Mitigation
# ===========================================================================
elif page == "🛠️ Bias Mitigation":
    st.title("🛠️ Bias Mitigation & Model Correction")
    st.markdown(
        "Apply fairness-aware reweighting to reduce gender bias in the hiring model. "
        "Compare fairness metrics before and after mitigation."
    )

    metrics_before = compute_all_metrics(df)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("⚡ Apply Mitigation", use_container_width=True, type="primary"):
            with st.spinner("Applying fairness-aware reweighting..."):
                st.session_state.df_mitigated = generate_mitigated_dataset(df)
            st.success("✅ Mitigation applied. Scroll down to see the comparison.")

    with col_info:
        st.info(
            "**Technique:** Fairness-aware sample reweighting. "
            "The model is retrained without the discriminatory penalty terms "
            "(gender-coded keywords, university type), relying only on merit-based features."
        )

    if st.session_state.df_mitigated is not None:
        df_mit = st.session_state.df_mitigated
        metrics_after = compute_all_metrics(df_mit)

        st.divider()
        st.subheader("Before vs After Mitigation")

        c1, c2, c3 = st.columns(3)

        dir_before = metrics_before["disparate_impact_ratio"]
        dir_after = metrics_after["disparate_impact_ratio"]
        dir_improvement = ((dir_after - dir_before) / max(dir_before, 0.001)) * 100

        bs_before = metrics_before["bias_score"]
        bs_after = metrics_after["bias_score"]

        dpd_before = metrics_before["demographic_parity_difference"]
        dpd_after = metrics_after["demographic_parity_difference"]

        with c1:
            st.metric(
                "Disparate Impact Ratio",
                f"{dir_after:.3f}",
                delta=f"+{dir_improvement:.1f}% improvement",
                delta_color="normal",
            )
            status = "✅ PASS" if dir_after >= 0.8 else "❌ FAIL"
            st.markdown(f"Before: `{dir_before:.3f}` → After: `{dir_after:.3f}` {status}")

        with c2:
            st.metric(
                "Bias Score",
                f"{bs_after:.3f}",
                delta=f"{bs_after - bs_before:.3f}",
                delta_color="inverse",
            )
            st.markdown(f"Before: `{bs_before:.3f}` → After: `{bs_after:.3f}`")

        with c3:
            st.metric(
                "Demographic Parity Difference",
                f"{dpd_after:.3f}",
                delta=f"{dpd_after - dpd_before:.3f}",
                delta_color="inverse",
            )
            st.markdown(f"Before: `{dpd_before:.3f}` → After: `{dpd_after:.3f}`")

        st.divider()

        # Side-by-side hire rate comparison
        st.subheader("Hire Rate Comparison")
        compare_data = pd.DataFrame(
            {
                "Group": ["Male (Before)", "Female (Before)", "Male (After)", "Female (After)"],
                "Hire Rate": [
                    metrics_before["male_hire_rate"],
                    metrics_before["female_hire_rate"],
                    metrics_after["male_hire_rate"],
                    metrics_after["female_hire_rate"],
                ],
                "Phase": ["Before", "Before", "After", "After"],
                "Gender": ["Male", "Female", "Male", "Female"],
            }
        )
        compare_chart = (
            alt.Chart(compare_data)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Phase:N", title=None),
                y=alt.Y("Hire Rate:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Gender:N",
                    scale=alt.Scale(domain=["Male", "Female"], range=["#60a5fa", "#f472b6"]),
                ),
                column=alt.Column("Gender:N", title=None),
                tooltip=[
                    alt.Tooltip("Phase:N"),
                    alt.Tooltip("Gender:N"),
                    alt.Tooltip("Hire Rate:Q", format=".1%"),
                ],
            )
            .properties(width=200, height=280)
        )
        st.altair_chart(compare_chart)

        if dir_after >= 0.8:
            st.success(
                f"🎉 After mitigation, the Disparate Impact Ratio is {dir_after:.3f} — "
                "above the 0.80 threshold. The model is now compliant with the four-fifths rule."
            )
        else:
            st.warning(
                f"⚠️ After mitigation, the DIR is {dir_after:.3f} — still below 0.80. "
                "Further intervention may be required."
            )
    else:
        st.info("Click **Apply Mitigation** above to see the before/after comparison.")


# ===========================================================================
# PAGE: Applicant View
# ===========================================================================
elif page == "👤 Applicant View":
    st.title("👤 Applicant View")
    st.markdown(
        "Job applicants can look up their hiring outcome and receive a plain-language "
        "explanation of the decision. If bias factors are detected, the decision is "
        "flagged as eligible for human review."
    )

    applicant_id_input = st.text_input(
        "Enter your Applicant ID (e.g. APP-0001)",
        value="APP-0001",
        max_chars=10,
    )

    if applicant_id_input:
        match = df[df["applicant_id"] == applicant_id_input.strip().upper()]
        if len(match) == 0:
            st.error(f"No applicant found with ID '{applicant_id_input}'. Please check and try again.")
        else:
            row = match.iloc[0]
            st.divider()

            # Outcome display
            if row["hired"] == 1:
                st.success("## ✅ Outcome: HIRED")
            else:
                st.error("## ❌ Outcome: NOT HIRED")

            st.markdown(f"**Overall Score:** `{row['model_score']:.4f}` out of 1.00")

            st.divider()
            st.subheader("Why was this decision made?")
            st.markdown("Here are the top three factors that influenced your result:")

            explanations = plain_english_explanation(row, df, top_n=3)
            for i, exp in enumerate(explanations, 1):
                st.markdown(f"**{i}.** {exp}")

            # Bias flag check
            has_bias_factor = row["gender_coded_keywords"] > 0 or row["womens_college"] == 1
            if has_bias_factor:
                st.divider()
                st.warning(
                    "### ⚠️ Potential Bias Detected\n\n"
                    "Our audit system has identified that one or more factors in your application "
                    "may have introduced bias:\n\n"
                    + (f"- Your resume contained **{int(row['gender_coded_keywords'])} gender-coded keyword(s)**.\n" if row["gender_coded_keywords"] > 0 else "")
                    + ("- You attended an **all-women's college**, which was penalised by the model.\n" if row["womens_college"] == 1 else "")
                    + "\nThis decision is **eligible for human review**. "
                    "You may request a manual review by clicking the button below."
                )

                already_queued = any(
                    r["applicant_id"] == applicant_id_input.strip().upper()
                    for r in st.session_state.review_queue
                )

                if already_queued:
                    st.info("✅ This application has already been flagged for human review.")
                else:
                    if st.button("🚩 Flag for Human Review", type="primary"):
                        st.session_state.review_queue.append(
                            {
                                "applicant_id": applicant_id_input.strip().upper(),
                                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                                "reason": "Bias factors detected (gender-coded keywords / all-women's college)",
                            }
                        )
                        st.success(
                            "✅ Your application has been flagged for human review. "
                            "A recruiter will manually assess your case within 5 business days."
                        )
            else:
                st.info("✅ No bias factors were detected in this application.")

    # Review queue display (for demo purposes)
    if st.session_state.review_queue:
        st.divider()
        st.subheader("📋 Human Review Queue (Demo)")
        queue_df = pd.DataFrame(st.session_state.review_queue)
        st.dataframe(queue_df, use_container_width=True)


# ===========================================================================
# PAGE: Audit Report
# ===========================================================================
elif page == "📋 Audit Report":
    st.title("📋 Audit & Compliance Report")
    st.markdown(
        "Generate a structured fairness audit report for regulatory compliance. "
        "The report documents all bias metrics, dataset statistics, and mitigation results."
    )

    metrics = compute_all_metrics(df)
    dir_val = metrics["disparate_impact_ratio"]
    is_compliant = dir_val >= 0.8

    # Compliance status banner
    if is_compliant:
        st.success("## ✅ Compliance Status: COMPLIANT")
        st.markdown(
            f"The Disparate Impact Ratio ({dir_val:.3f}) meets the four-fifths rule threshold (≥ 0.80)."
        )
    else:
        st.error("## ❌ Compliance Status: NON-COMPLIANT")
        st.markdown(
            f"The Disparate Impact Ratio ({dir_val:.3f}) is below the four-fifths rule threshold (0.80). "
            "Immediate action is required."
        )

    st.divider()

    # Summary statistics
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants", f"{metrics['total_applicants']:,}")
    col2.metric("Total Hired", f"{metrics['total_hired']:,}")
    col3.metric(
        "Overall Hire Rate",
        f"{metrics['total_hired'] / metrics['total_applicants']:.1%}",
    )

    gender_summary = (
        df.groupby("gender")
        .agg(
            count=("applicant_id", "count"),
            hired=("hired", "sum"),
            hire_rate=("hired", "mean"),
            avg_score=("model_score", "mean"),
            avg_keywords=("gender_coded_keywords", "mean"),
        )
        .reset_index()
    )
    gender_summary.columns = [
        "Gender", "Applicants", "Hired", "Hire Rate", "Avg Score", "Avg Gender Keywords"
    ]
    gender_summary["Hire Rate"] = gender_summary["Hire Rate"].map("{:.1%}".format)
    gender_summary["Avg Score"] = gender_summary["Avg Score"].map("{:.4f}".format)
    gender_summary["Avg Gender Keywords"] = gender_summary["Avg Gender Keywords"].map("{:.2f}".format)
    st.dataframe(gender_summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Fairness Metrics")

    metrics_display = {
        "Metric": [
            "Disparate Impact Ratio",
            "Demographic Parity Difference",
            "Equalised Odds TPR Gap",
            "Overall Bias Score",
            "Male Hire Rate",
            "Female Hire Rate",
        ],
        "Value": [
            f"{metrics['disparate_impact_ratio']:.4f}",
            f"{metrics['demographic_parity_difference']:.4f}",
            f"{metrics['equalised_odds_tpr_gap']:.4f}",
            f"{metrics['bias_score']:.4f}",
            f"{metrics['male_hire_rate']:.4f}",
            f"{metrics['female_hire_rate']:.4f}",
        ],
        "Threshold": ["≥ 0.80", "≈ 0.00", "≈ 0.00", "< 0.30", "—", "—"],
        "Status": [
            "✅ PASS" if metrics["disparate_impact_ratio"] >= 0.8 else "❌ FAIL",
            "✅ PASS" if abs(metrics["demographic_parity_difference"]) < 0.05 else "❌ FAIL",
            "✅ PASS" if abs(metrics["equalised_odds_tpr_gap"]) < 0.05 else "❌ FAIL",
            "✅ LOW" if metrics["bias_score"] < 0.3 else ("⚠️ MODERATE" if metrics["bias_score"] < 0.5 else "❌ HIGH"),
            "—",
            "—",
        ],
    }
    st.dataframe(pd.DataFrame(metrics_display), use_container_width=True, hide_index=True)

    # Mitigation results if available
    if st.session_state.df_mitigated is not None:
        st.divider()
        st.subheader("Mitigation Results")
        m_after = compute_all_metrics(st.session_state.df_mitigated)
        mit_display = {
            "Metric": ["Disparate Impact Ratio", "Bias Score", "Demographic Parity Difference"],
            "Before": [
                f"{metrics['disparate_impact_ratio']:.4f}",
                f"{metrics['bias_score']:.4f}",
                f"{metrics['demographic_parity_difference']:.4f}",
            ],
            "After": [
                f"{m_after['disparate_impact_ratio']:.4f}",
                f"{m_after['bias_score']:.4f}",
                f"{m_after['demographic_parity_difference']:.4f}",
            ],
        }
        st.dataframe(pd.DataFrame(mit_display), use_container_width=True, hide_index=True)

    st.divider()

    # Download button
    st.subheader("Download Audit Report")

    def build_csv_report():
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        lines = [
            "FairSight Audit Report",
            f"Generated: {timestamp}",
            f"Compliance Status: {'COMPLIANT' if is_compliant else 'NON-COMPLIANT'}",
            "",
            "Dataset Summary",
            f"Total Applicants,{metrics['total_applicants']}",
            f"Total Hired,{metrics['total_hired']}",
            f"Male Hire Rate,{metrics['male_hire_rate']:.4f}",
            f"Female Hire Rate,{metrics['female_hire_rate']:.4f}",
            "",
            "Fairness Metrics",
            f"Disparate Impact Ratio,{metrics['disparate_impact_ratio']:.4f}",
            f"Demographic Parity Difference,{metrics['demographic_parity_difference']:.4f}",
            f"Equalised Odds TPR Gap,{metrics['equalised_odds_tpr_gap']:.4f}",
            f"Overall Bias Score,{metrics['bias_score']:.4f}",
        ]
        if st.session_state.df_mitigated is not None:
            m_after = compute_all_metrics(st.session_state.df_mitigated)
            lines += [
                "",
                "Post-Mitigation Metrics",
                f"Disparate Impact Ratio (After),{m_after['disparate_impact_ratio']:.4f}",
                f"Bias Score (After),{m_after['bias_score']:.4f}",
                f"Demographic Parity Difference (After),{m_after['demographic_parity_difference']:.4f}",
            ]
        return "\n".join(lines)

    csv_content = build_csv_report()
    st.download_button(
        label="⬇️ Download Audit Report (CSV)",
        data=csv_content,
        file_name=f"fairsight_audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Human review queue
    if st.session_state.review_queue:
        st.divider()
        st.subheader("Human Review Queue")
        queue_df = pd.DataFrame(st.session_state.review_queue)
        st.dataframe(queue_df, use_container_width=True, hide_index=True)
        queue_csv = queue_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Review Queue (CSV)",
            data=queue_csv,
            file_name="fairsight_review_queue.csv",
            mime="text/csv",
        )

