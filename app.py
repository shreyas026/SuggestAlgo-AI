"""
SuggestAlgo AI — Explainable AI Assistant for Algorithm Selection

Dual Mode Application:
MODE 1: Dataset / CSV-based Machine Learning Algorithm Recommendation & Benchmarking
MODE 2: Natural-Language Problem & Project Idea Algorithm Recommendation Engine

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
from src.model_evaluation import evaluate_all_models
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
</style>
""", unsafe_allow_html=True)


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
# MODE 1 — DATASET / CSV WORKFLOW
# ============================================================================

def render_mode_1_dataset():
    st.subheader("📊 Mode 1: Dataset / CSV Algorithm Selection & ML Benchmarking")
    st.markdown("*Upload a CSV dataset or choose a pre-loaded academic dataset. SuggestAlgo AI will profile meta-features, recommend candidate algorithms via meta-learning, benchmark 9 models with 5-Fold Stratified Cross-Validation, and provide SHAP explanations.*")
    
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

        # Run pipeline when clicked or if session has results
        if run_btn or 'mode1_results' not in st.session_state or st.session_state.get('last_dataset') != dataset_name:
            with st.spinner("Processing dataset through ML pipeline (Profiling → Preprocessing → Meta-Learning → 5-Fold CV Benchmarking)..."):
                profile_res = profile_dataset(df)
                X_proc, y_proc, feat_names, label_enc, prep_info = preprocess_dataset(df, target_col)
                meta_feats = extract_meta_features(df, target_col)
                rec_res = get_meta_learning_recommendation(meta_feats)
                results_dict, comparison_df, best_model_name, split_data = evaluate_all_models(
                    X_proc, y_proc, recommended_algo=rec_res['recommended_algo_raw']
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
                    'df': df
                }
                st.session_state.last_dataset = dataset_name

        res = st.session_state.mode1_results

        # Mode 1 Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dataset Profile",
            "🎯 Algorithm Recommendation",
            "📈 Model Evaluation",
            "🧠 Explainable AI (SHAP)",
            "🔬 ML Proof & Export"
        ])
        
        # TAB 1: DATASET PROFILE
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

        # TAB 2: ALGORITHM RECOMMENDATION
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

        # TAB 3: MODEL EVALUATION
        with tab3:
            st.subheader("Candidate Model Benchmarking (9 Algorithms)")
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

        # TAB 4: EXPLAINABLE AI
        with tab4:
            st.subheader(f"SHAP Model Explanations — Best Model ({res['best_model_name']})")
            best_model = res['results_dict'][res['best_model_name']]['trained_model']
            X_train = res['split_data']['X_train']
            X_test = res['split_data']['X_test']
            
            explainer, exp_type = get_shap_explainer(best_model, X_train, res['best_model_name'])
            if explainer is not None:
                shap_vals, X_samp = compute_shap_values(explainer, X_test, exp_type, res['best_model_name'])
                fig_shap = plot_shap_summary(shap_vals, X_samp, res['feat_names'], res['best_model_name'])
                if fig_shap:
                    st.pyplot(fig_shap)
                else:
                    st.info("SHAP plot generated using scikit-learn feature importances below.")
            
            feat_imp = get_feature_importance(best_model, res['feat_names'], res['best_model_name'])
            if feat_imp:
                fig_imp = plot_feature_importance(feat_imp, f"Feature Importance — {res['best_model_name']}")
                if fig_imp:
                    st.pyplot(fig_imp)

        # TAB 5: ML PROOF & EXPORT
        with tab5:
            st.subheader("🔬 Academic ML Proof & Cross-Validation Results")
            st.markdown("""
            > **Verification Note**: The >85% accuracy target applies to the included demonstration datasets (Iris, Breast Cancer, Wine) and is verified via empirical 5-Fold Stratified Cross-Validation.
            """)
            
            st.markdown("### 5-Fold Stratified Cross Validation Benchmark")
            st.dataframe(res['comparison_df'], use_container_width=True)
            
            # CSV Download
            csv_data = res['comparison_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download ML Results CSV Report",
                data=csv_data,
                file_name=f"SuggestAlgo_ML_Results_{res['dataset_name']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )


# ============================================================================
# MODE 2 — PROBLEM / PROJECT IDEA WORKFLOW
# ============================================================================

def render_mode_2_problem():
    st.subheader("💡 Mode 2: Problem / Question / Project Idea Algorithm Recommendation")
    st.markdown("*Describe your coding problem, algorithmic query, or AI project idea in natural language. SuggestAlgo AI will analyze requirements using semantic embeddings, map them against a rich algorithm knowledge base, and provide structured recommendations, complexity analysis, and Python templates.*")
    
    # Preset Example Buttons
    st.markdown("**Quick Test Examples:**")
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    
    prompt_text = ""
    if ex_col1.button("🔍 Binary Search (Sorted Array)", use_container_width=True):
        prompt_text = "I have a list of one million sorted numbers and I need to find whether a particular number exists as efficiently as possible."
    if ex_col2.button("🚗 Shortest Path (City Graph)", use_container_width=True):
        prompt_text = "I want to find the shortest distance and optimal path between two cities in a weighted graph network."
    if ex_col3.button("📉 Customer Churn (ML)", use_container_width=True):
        prompt_text = "I want to build a machine learning model that predicts whether customers will leave a company based on their usage history."
    if ex_col4.button("🎬 Movie Recommender", use_container_width=True):
        prompt_text = "I want to build an AI system that recommends movies to users based on what similar users watched."

    user_input = st.text_area(
        "Describe your problem or project idea in detail:",
        value=prompt_text,
        height=140,
        placeholder="Example: I have a sorted array with 1,000,000 items and need to check if a target item exists efficiently. What algorithm should I use?"
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
        st.markdown(f"**Category**: `{top_algo['category']}` | **Expected Time Complexity**: `{top_algo['time_complexity']}` | **Space Complexity**: `{top_algo['space_complexity']}`")
        
        # 2. PROBLEM UNDERSTANDING & EXTRACTION
        with st.expander("🔍 Problem Understanding & Requirements Extraction", expanded=True):
            p1, p2, p3 = st.columns(3)
            p1.write(f"**Detected Problem Category**: {analysis['category']}")
            p2.write(f"**Target Data Structure**: {analysis['data_structure']}")
            p3.write(f"**Problem Type**: {'Machine Learning / AI Project' if analysis['is_ml_project'] else 'Algorithmic / Computational'}")
            
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


if __name__ == "__main__":
    main()
