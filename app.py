"""
SuggestAlgo AI — Explainable AI Assistant for Algorithm Selection

Dual Mode Application:
MODE 1: Dataset / CSV-based Machine Learning Algorithm Recommendation & Benchmarking
         Supports both Classification and Regression tasks (auto-detected).
MODE 2: Natural-Language Problem & Project Idea Algorithm Recommendation Engine
         Supports text input and optional image/photo input (OCR-based problem extraction).

Foundation / Reference: AMLBID by LeMGarouani et al.
GitHub: https://github.com/LeMGarouani/AMLBID
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset, validate_file, detect_column_types, get_target_candidates, validate_target
from src.preprocessing import preprocess_dataset
from src.profiling import profile_dataset, format_profile_for_display
from src.algorithm_selection import extract_meta_features, get_meta_learning_recommendation, get_algorithm_name_mapping
from src.model_evaluation import evaluate_all_models, detect_problem_type
from src.explainability import (
    get_shap_explainer, compute_shap_values, get_feature_importance,
    plot_feature_importance, plot_shap_summary, plot_shap_beeswarm,
    explain_algorithm_selection
)
from src.utils import get_sample_datasets, load_sample_dataset, get_model_description
from src.nlp_recommendation import recommend_algorithm_from_text

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SuggestAlgo AI — Explainable Algorithm Selection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #4B5563;
        font-size: 1.15rem;
    }
    .mode-card {
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: #F9FAFB;
        transition: all 0.3s ease;
    }
    .mode-card:hover {
        border-color: #6366F1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        color: #374151;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #1F2937;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .task-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .task-classification {
        background: #DBEAFE;
        color: #1D4ED8;
    }
    .task-regression {
        background: #D1FAE5;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER: OCR / IMAGE READING
# ============================================================================

def extract_text_from_image(uploaded_image):
    """
    Try to extract text from an uploaded image using pytesseract (OCR).
    Falls back to a descriptive placeholder if pytesseract is not installed.
    """
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(uploaded_image)
        text = pytesseract.image_to_string(img).strip()
        if text:
            return text, True
        return "", False
    except ImportError:
        # pytesseract not installed — return placeholder
        return "", False
    except Exception:
        return "", False


# ============================================================================
# HELPER FUNCTIONS & RENDERING
# ============================================================================

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 SuggestAlgo AI</h1>
        <p>Explainable AI System for Machine Learning & Computational Algorithm Selection</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    render_header()

    # Initialize session state for mode selection
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'HOME'

    # Navigation Buttons on Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode_choice = st.radio(
            "Select Recommendation Mode:",
            ["📊 Mode 1 — Dataset / CSV", "💡 Mode 2 — Problem / Project Idea"],
            index=0 if st.session_state.app_mode != 'MODE_2' else 1,
            horizontal=True
        )
        if "Mode 1" in mode_choice:
            st.session_state.app_mode = 'MODE_1'
        else:
            st.session_state.app_mode = 'MODE_2'

    st.markdown("---")

    if st.session_state.app_mode == 'MODE_1':
        render_mode_1_dataset()
    else:
        render_mode_2_problem()


# ============================================================================
# MODE 1 — DATASET / CSV WORKFLOW (Classification + Regression)
# ============================================================================

def render_mode_1_dataset():
    st.subheader("📊 Mode 1: Dataset / CSV Algorithm Selection & ML Benchmarking")
    st.markdown(
        "*Upload a CSV dataset or choose a pre-loaded academic dataset. "
        "SuggestAlgo AI will auto-detect the task type (Classification or Regression), "
        "profile meta-features, recommend algorithms via meta-learning, benchmark models "
        "with 5-Fold Cross-Validation, and provide SHAP explanations.*"
    )

    with st.sidebar:
        st.header("⚙️ Mode 1 Settings")
        data_source = st.radio("Dataset Source", ["Sample Datasets", "Upload CSV File"])

        df = None
        dataset_name = ""

        if data_source == "Sample Datasets":
            sample_options = get_sample_datasets()
            selected_sample = st.selectbox(
                "Choose Sample Dataset",
                options=list(sample_options.keys()),
                format_func=lambda x: f"{x} ({sample_options[x]['description']})"
            )
            df = load_sample_dataset(selected_sample)
            dataset_name = selected_sample
            target_candidates = get_target_candidates(df)
            default_target = sample_options[selected_sample]['target']
            target_idx = target_candidates.index(default_target) if default_target in target_candidates else 0
            target_col = st.selectbox("Select Target Column", target_candidates, index=target_idx)

        else:
            uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
            if uploaded_file is not None:
                df, err = load_dataset(uploaded_file)
                if err:
                    st.error(f"Error loading CSV: {err}")
                    return
                dataset_name = uploaded_file.name
                target_candidates = get_target_candidates(df)
                target_col = st.selectbox("Select Target Column", target_candidates)
            else:
                st.info("Please upload a CSV file or switch to Sample Datasets.")
                return

        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if df is not None and target_col:
        valid, msg = validate_target(df, target_col)
        if not valid:
            st.warning(f"Target validation issue: {msg}")

        # Detect problem type early (before preprocessing) for display
        raw_y = df[target_col].dropna()
        detected_type = detect_problem_type(raw_y.values if hasattr(raw_y, 'values') else raw_y)

        badge_class = "task-classification" if detected_type == "classification" else "task-regression"
        badge_label = "🔵 Classification" if detected_type == "classification" else "🟢 Regression"
        st.markdown(
            f'<span class="task-badge {badge_class}">Detected Task: {badge_label}</span>',
            unsafe_allow_html=True
        )

        # Run pipeline when clicked or dataset changed
        if run_btn or 'mode1_results' not in st.session_state or \
                st.session_state.get('last_dataset') != dataset_name:
            with st.spinner(
                "Processing dataset through ML pipeline "
                "(Profiling → Preprocessing → Meta-Learning → CV Benchmarking)..."
            ):
                profile_res = profile_dataset(df)
                X_proc, y_proc, feat_names, label_enc, prep_info = preprocess_dataset(
                    df, target_col, problem_type=detected_type
                )
                meta_feats = extract_meta_features(df, target_col)

                # get_meta_learning_recommendation returns (algo_name, confidence_info)
                rec_algo, rec_conf = get_meta_learning_recommendation(meta_feats)
                algo_map = get_algorithm_name_mapping()
                rec_res = {
                    'recommended_algo': algo_map.get(rec_algo, rec_algo),
                    'recommended_algo_raw': rec_algo,
                    'confidence': rec_conf['confidence'],
                    'explanation': (
                        f"Meta-learning KNN identified '{algo_map.get(rec_algo, rec_algo)}' as the "
                        f"best algorithm based on {rec_conf['confidence']:.1f}% weighted neighbor agreement "
                        f"across {len(rec_conf['neighbor_algorithms'])} nearest datasets in the knowledge base."
                    ),
                    'meta_features_used': rec_conf['meta_features_used'],
                    'neighbor_algorithms': rec_conf['neighbor_algorithms'],
                }

                results_dict, comparison_df, best_model_name, split_data = evaluate_all_models(
                    X_proc, y_proc,
                    recommended_algo=rec_res['recommended_algo_raw'],
                    problem_type=detected_type
                )

                st.session_state.mode1_results = {
                    'profile_res': profile_res,
                    'X_proc': X_proc,
                    'y_proc': y_proc,
                    'feat_names': feat_names,
                    'label_enc': label_enc,
                    'prep_info': prep_info,
                    'meta_feats': meta_feats,
                    'rec_res': rec_res,
                    'results_dict': results_dict,
                    'comparison_df': comparison_df,
                    'best_model_name': best_model_name,
                    'split_data': split_data,
                    'dataset_name': dataset_name,
                    'df': df,
                    'problem_type': detected_type
                }
                st.session_state.last_dataset = dataset_name

        res = st.session_state.mode1_results
        problem_type = res.get('problem_type', 'classification')

        # Mode 1 Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dataset Profile",
            "🎯 Algorithm Recommendation",
            "📈 Model Evaluation",
            "🧠 Explainable AI (SHAP)",
            "🔬 ML Proof & Export"
        ])

        # ── TAB 1: DATASET PROFILE ──────────────────────────────────────────
        with tab1:
            st.subheader(f"Dataset Overview — {res['dataset_name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", res['profile_res']['shape'][0])
            c2.metric("Columns", res['profile_res']['shape'][1])
            c3.metric("Numerical Features", len(res['profile_res']['numerical_cols']))
            c4.metric("Categorical Features", len(res['profile_res']['categorical_cols']))

            st.markdown("### Raw Data Preview")
            st.dataframe(res['df'].head(10), use_container_width=True)

            st.markdown("### Summary Statistics")
            st.dataframe(res['df'].describe(), use_container_width=True)

        # ── TAB 2: ALGORITHM RECOMMENDATION ────────────────────────────────
        with tab2:
            st.subheader("Meta-Learning Algorithm Recommendation")
            rec = res['rec_res']

            rc1, rc2 = st.columns([2, 1])
            with rc1:
                st.success(f"### 💡 Recommended Algorithm: **{rec['recommended_algo']}**")
                st.write(rec['explanation'])
            with rc2:
                st.metric("Meta-Learning Confidence", f"{rec['confidence']:.1f}%")
                st.write("**Top Meta-Feature Matches:**")
                for k, v in list(rec['meta_features_used'].items())[:5]:
                    st.write(f"- `{k}`: {v:.4f}" if isinstance(v, float) else f"- `{k}`: {v}")

        # ── TAB 3: MODEL EVALUATION ─────────────────────────────────────────
        with tab3:
            n_models = len(res['results_dict'])
            if problem_type == 'regression':
                st.subheader(f"Regression Candidate Model Benchmarking ({n_models} Algorithms)")
                st.info(
                    "**Regression Task Detected** — Metrics shown: MAE (↓ lower is better), "
                    "RMSE (↓), R² Score (↑ closer to 1.0 is better), 5-Fold CV R² Mean."
                )
                st.dataframe(res['comparison_df'], use_container_width=True)

                if 'R² Score' in res['comparison_df'].columns:
                    fig = px.bar(
                        res['comparison_df'],
                        x='R² Score',
                        y='Algorithm',
                        orientation='h',
                        title="R² Score Comparison Across Regression Candidates",
                        color='R² Score',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fig2 = px.bar(
                        res['comparison_df'],
                        x='RMSE',
                        y='Algorithm',
                        orientation='h',
                        title="RMSE Comparison (Lower is Better)",
                        color='RMSE',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.subheader("Classification Candidate Model Benchmarking (9 Algorithms)")
                st.dataframe(res['comparison_df'], use_container_width=True)

                fig = px.bar(
                    res['comparison_df'],
                    x='F1 Score',
                    y='Algorithm',
                    orientation='h',
                    title="F1 Score Comparison Across Candidates",
                    color='F1 Score',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── TAB 4: EXPLAINABLE AI ────────────────────────────────────────────
        with tab4:
            st.subheader(f"SHAP Model Explanations — Best Model ({res['best_model_name']})")

            if problem_type == 'regression':
                # SHAP works for regression tree models too
                best_model = res['results_dict'][res['best_model_name']]['trained_model']
                X_train = res['split_data']['X_train']
                X_test = res['split_data']['X_test']

                # Try tree-based SHAP for regression models
                regression_tree_models = ['RandomForest', 'DecisionTree', 'ExtraTrees',
                                           'GradientBoosting', 'XGBoost']
                if res['best_model_name'] in regression_tree_models:
                    try:
                        import shap
                        explainer = shap.TreeExplainer(best_model)
                        X_samp = X_test[:min(200, len(X_test))]
                        shap_vals = explainer.shap_values(X_samp)
                        if shap_vals is not None:
                            fig_shap, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_vals, pd.DataFrame(X_samp, columns=res['feat_names']),
                                              show=False, plot_type="dot")
                            st.pyplot(plt.gcf())
                            plt.close('all')
                    except Exception as shap_err:
                        st.info(f"SHAP tree plot unavailable for this model: {shap_err}")
                else:
                    st.info("SHAP plots are shown for tree-based models. Showing feature importances below.")

                # Feature importance for all regression models
                feat_imp = get_feature_importance(best_model, res['feat_names'], res['best_model_name'])
                if feat_imp:
                    fig_imp = plot_feature_importance(feat_imp, f"Feature Importance — {res['best_model_name']}")
                    if fig_imp:
                        st.pyplot(fig_imp)
            else:
                best_model = res['results_dict'][res['best_model_name']]['trained_model']
                X_train = res['split_data']['X_train']
                X_test = res['split_data']['X_test']

                explainer, exp_type = get_shap_explainer(best_model, X_train, res['best_model_name'])
                if explainer is not None:
                    shap_vals, X_samp = compute_shap_values(
                        explainer, X_test, exp_type, res['best_model_name']
                    )
                    fig_shap = plot_shap_summary(shap_vals, X_samp, res['feat_names'], res['best_model_name'])
                    if fig_shap:
                        st.pyplot(fig_shap)
                    else:
                        st.info("SHAP plot generated using scikit-learn feature importances below.")

                feat_imp = get_feature_importance(best_model, res['feat_names'], res['best_model_name'])
                if feat_imp:
                    fig_imp = plot_feature_importance(
                        feat_imp, f"Feature Importance — {res['best_model_name']}"
                    )
                    if fig_imp:
                        st.pyplot(fig_imp)

        # ── TAB 5: ML PROOF & EXPORT ─────────────────────────────────────────
        with tab5:
            st.subheader("🔬 Academic ML Proof & Cross-Validation Results")

            if problem_type == 'regression':
                st.markdown("""
                > **Verification Note**: Regression results are verified via empirical
                > 5-Fold KFold Cross-Validation. Primary ranking metric: R² Score (coefficient of determination).
                > Additional metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error).
                """)
                st.markdown("### 5-Fold KFold Cross Validation Benchmark — Regression")
            else:
                st.markdown("""
                > **Verification Note**: The >85% accuracy target applies to the included
                > demonstration datasets (Iris, Breast Cancer, Wine) and is verified via
                > empirical 5-Fold Stratified Cross-Validation.
                """)
                st.markdown("### 5-Fold Stratified Cross Validation Benchmark — Classification")

            st.dataframe(res['comparison_df'], use_container_width=True)

            # CSV Download
            csv_data = res['comparison_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download ML Results CSV Report",
                data=csv_data,
                file_name=(
                    f"SuggestAlgo_ML_Results_{res['dataset_name']}_"
                    f"{res['problem_type']}_{datetime.now().strftime('%Y%m%d')}.csv"
                ),
                mime="text/csv",
                type="primary"
            )


# ============================================================================
# MODE 2 — PROBLEM / PROJECT IDEA WORKFLOW (Text + Optional Image Input)
# ============================================================================

def render_mode_2_problem():
    st.subheader("💡 Mode 2: Problem / Question / Project Idea Algorithm Recommendation")
    st.markdown(
        "*Describe your coding problem, algorithmic query, or AI project idea in natural language "
        "— or upload an image/photo of a handwritten or printed problem statement. "
        "SuggestAlgo AI will analyze requirements using semantic embeddings, map them against a "
        "rich algorithm knowledge base, and provide structured recommendations, complexity analysis, "
        "and Python templates.*"
    )

    # ── Image Input ──────────────────────────────────────────────────────────
    with st.expander("📷 Optional: Upload Image / Photo of Problem Statement", expanded=False):
        uploaded_image = st.file_uploader(
            "Upload an image (JPG, PNG, etc.) of a handwritten or printed problem:",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            key="mode2_image_upload"
        )

        ocr_text = ""
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Problem Image", use_column_width=True)
            with st.spinner("Extracting text from image..."):
                ocr_text, ocr_success = extract_text_from_image(uploaded_image)

            if ocr_success and ocr_text.strip():
                st.success("✅ Text successfully extracted from image via OCR:")
                st.code(ocr_text, language="text")
                st.info("The extracted text has been pre-filled in the problem description below.")
            else:
                st.warning(
                    "⚠️ Could not automatically extract text from this image. "
                    "Please type your problem description manually below. "
                    "(Install `pytesseract` + Tesseract OCR for automatic image-to-text support.)"
                )

    # ── Preset Example Buttons ────────────────────────────────────────────────
    st.markdown("**Quick Test Examples:**")
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

    prompt_text = ocr_text  # Pre-fill from OCR if available

    if ex_col1.button("🔍 Binary Search (Sorted Array)", use_container_width=True):
        prompt_text = "I have a list of one million sorted numbers and I need to find whether a particular number exists as efficiently as possible."
    if ex_col2.button("🚗 Shortest Path (City Graph)", use_container_width=True):
        prompt_text = "I want to find the shortest distance and optimal path between two cities in a weighted graph network."
    if ex_col3.button("📉 Customer Churn (ML)", use_container_width=True):
        prompt_text = "I want to build a machine learning model that predicts whether customers will leave a company based on their usage history."
    if ex_col4.button("🎬 Movie Recommender", use_container_width=True):
        prompt_text = "I want to build an AI system that recommends movies to users based on what similar users watched."

    # Additional examples row
    ex_col5, ex_col6, ex_col7, ex_col8 = st.columns(4)
    if ex_col5.button("🏠 House Price Prediction", use_container_width=True):
        prompt_text = "I want to predict house prices based on features like location, size, number of rooms, and age of the property."
    if ex_col6.button("🔗 Merge Sort (Large Data)", use_container_width=True):
        prompt_text = "I have a very large unsorted array and need to sort it efficiently in O(n log n) time."
    if ex_col7.button("🕵️ Fraud Detection", use_container_width=True):
        prompt_text = "I want to detect fraudulent credit card transactions in a highly imbalanced dataset."
    if ex_col8.button("🌐 Web Crawler (BFS)", use_container_width=True):
        prompt_text = "I need to crawl all pages of a website starting from the root page and visit every linked page exactly once."

    user_input = st.text_area(
        "Describe your problem or project idea in detail:",
        value=prompt_text,
        height=140,
        placeholder=(
            "Example: I have a sorted array with 1,000,000 items and need to check if a target "
            "item exists efficiently. What algorithm should I use?"
        )
    )

    if st.button("🚀 Analyze & Recommend Algorithm", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a problem description or project idea.")
            return

        with st.spinner("Analyzing problem requirements & computing semantic embeddings..."):
            rec_data = recommend_algorithm_from_text(user_input)

        analysis = rec_data['analysis']
        top_algo = rec_data['top_recommendation']
        top_score = rec_data['top_score']
        candidates = rec_data['candidates']

        st.markdown("---")

        # 1. TOP RECOMMENDATION HIGHLIGHT
        st.success(f"### 🏆 Recommended Algorithm: **{top_algo['name']}** (Confidence: {top_score}%)")
        st.markdown(
            f"**Category**: `{top_algo['category']}` | "
            f"**Expected Time Complexity**: `{top_algo['time_complexity']}` | "
            f"**Space Complexity**: `{top_algo['space_complexity']}`"
        )

        # 2. PROBLEM UNDERSTANDING & EXTRACTION
        with st.expander("🔍 Problem Understanding & Requirements Extraction", expanded=True):
            p1, p2, p3 = st.columns(3)
            p1.write(f"**Detected Problem Category**: {analysis['category']}")
            p2.write(f"**Target Data Structure**: {analysis['data_structure']}")
            p3.write(
                f"**Problem Type**: "
                f"{'Machine Learning / AI Project' if analysis['is_ml_project'] else 'Algorithmic / Computational'}"
            )

            if analysis['follow_up_questions']:
                st.info("💡 **Questions that would refine this recommendation further:**")
                for q in analysis['follow_up_questions']:
                    st.write(f"- {q}")

        # 3. WHY THIS ALGORITHM? (REASONING)
        st.markdown("### 💡 Why Was This Algorithm Recommended?")
        st.write(top_algo['description'])
        st.markdown("**Key Reasons for Match:**")
        for uc in top_algo['best_use_cases']:
            st.write(f"- ✅ **Applicability**: {uc}")
        for req in top_algo['requirements']:
            st.write(f"- ⚠️ **Requirement/Pre-condition**: {req}")

        # 4. ALGORITHM CANDIDATES COMPARISON TABLE
        st.markdown("### 📊 Top Algorithm Candidates & Complexity Comparison")
        comp_rows = []
        for item in candidates:
            a = item['algo']
            comp_rows.append({
                "Algorithm": a['name'],
                "Match Confidence": f"{item['similarity_score']}%",
                "Category": a['category'],
                "Time Complexity": a['time_complexity'],
                "Space Complexity": a['space_complexity'],
                "Best Use Cases": ", ".join(a['best_use_cases'][:2]),
                "Limitations": ", ".join(a['limitations'][:1])
            })
        st.table(pd.DataFrame(comp_rows))

        # 5. ALTERNATIVE ALGORITHMS & TRADE-OFFS
        st.markdown("### 🔄 Alternatives & Trade-Offs")
        st.write(f"**Main Alternatives to {top_algo['name']}**:")
        for alt in top_algo['alternatives']:
            st.write(f"- **{alt}**: Consider when specific constraints or data structures differ.")

        # 6. PYTHON IMPLEMENTATION TEMPLATE
        st.markdown("### 🐍 Python Implementation Guidance Template")
        st.code(top_algo['python_template'], language="python")

        # 7. IF ML PROJECT — SUGGEST SWITCHING TO MODE 1
        if analysis['is_ml_project']:
            st.markdown("---")
            st.info(
                "🔄 **This looks like an ML project!** "
                "Switch to **Mode 1 (Dataset/CSV)** to upload your actual dataset, "
                "run a full ML benchmarking pipeline with 9 algorithm candidates "
                "(classification) or 10 regressors, and get SHAP explanations on real data."
            )


if __name__ == "__main__":
    main()
