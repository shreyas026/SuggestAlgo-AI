"""
SuggestAlgo AI — Explainable AI for Machine Learning Algorithm Selection

A Streamlit-based academic ML application that:
1. Accepts a CSV dataset from the user
2. Profiles the dataset and extracts meta-features
3. Uses meta-learning (inspired by AMLBID) to recommend an ML algorithm
4. Trains and evaluates multiple candidate algorithms
5. Provides Explainable AI (SHAP) explanations

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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SuggestAlgo AI — Explainable Algorithm Selection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card h3 {
        color: #374151;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #1f2937;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    /* Recommendation card */
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .recommendation-card h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .recommendation-card .algo-name {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.8rem 0;
    }
    .recommendation-card .score {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    
    /* Section headers */
    .section-header {
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Status badges */
    .badge-success { background: #d1fae5; color: #065f46; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .badge-warning { background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .badge-info { background: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #9ca3af;
        font-size: 0.85rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #c7d2fe !important;
    }
    
    /* Hide default streamlit footer */
    footer { visibility: hidden; }
    
    /* Table styling */
    .dataframe { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# 🧠 SuggestAlgo AI")
    st.markdown("**Explainable AI for Algorithm Selection**")
    st.markdown("---")
    
    st.markdown("### 📋 Navigation")
    st.markdown("""
    1. 📂 Upload Dataset
    2. 📊 Dataset Profiling
    3. 🎯 Algorithm Selection
    4. 📈 Model Comparison
    5. 🔍 Explainability (XAI)
    6. 📋 Results Summary
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    Built on meta-learning principles from 
    [AMLBID](https://github.com/LeMGarouani/AMLBID) 
    by LeMGarouani et al.
    
    **Methodology:**
    - Meta-feature extraction
    - KNN-based algorithm selection
    - Multi-model evaluation
    - SHAP explainability
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    test_size = st.slider("Test Set Size (%)", 10, 50, 30, 5) / 100
    
    st.markdown("---")
    st.markdown(
        '<div style="color:#c7d2fe; font-size:0.75rem; text-align:center;">'
        'SuggestAlgo AI v1.0<br>Academic ML Project</div>',
        unsafe_allow_html=True
    )


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 SuggestAlgo AI</h1>
    <p>Explainable AI for Machine Learning Algorithm Selection</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION 1: DATASET UPLOAD
# ============================================================================

st.markdown('<div class="section-header"><h2>📂 Step 1: Load Dataset</h2></div>', unsafe_allow_html=True)

col_upload, col_sample = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=['csv'],
        help="Upload a CSV file with features and a target column for classification."
    )

with col_sample:
    st.markdown("**Or use a sample dataset:**")
    sample_datasets = get_sample_datasets()
    selected_sample = st.selectbox(
        "Select sample dataset",
        ["None"] + list(sample_datasets.keys()),
        help="Choose a built-in dataset for quick demonstration."
    )

# Load data
df = None
default_target = None

if uploaded_file is not None:
    validated_file, file_errors = validate_file(uploaded_file)
    if file_errors:
        for err in file_errors:
            st.error(f"⚠️ {err}")
    else:
        df, load_errors = load_dataset(uploaded_file)
        if load_errors:
            for err in load_errors:
                st.error(f"⚠️ {err}")
        elif df is not None:
            st.success(f"✅ Dataset loaded: **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

elif selected_sample != "None":
    dataset_key = sample_datasets[selected_sample]
    df, default_target = load_sample_dataset(dataset_key)
    if df is not None:
        st.success(f"✅ Sample dataset loaded: **{selected_sample}** — {df.shape[0]:,} rows × {df.shape[1]} columns")


# ============================================================================
# SECTION 2: TARGET SELECTION & DATASET PROFILING
# ============================================================================

if df is not None:
    st.markdown('<div class="section-header"><h2>🎯 Step 2: Select Target Column</h2></div>', unsafe_allow_html=True)
    
    # Detect good target candidates
    candidates = get_target_candidates(df)
    all_columns = df.columns.tolist()
    
    # Default to last column or known target
    if default_target and default_target in all_columns:
        default_idx = all_columns.index(default_target)
    elif candidates:
        default_idx = all_columns.index(candidates[0]) if candidates[0] in all_columns else len(all_columns) - 1
    else:
        default_idx = len(all_columns) - 1
    
    target_col = st.selectbox(
        "Select the target (label) column for classification:",
        all_columns,
        index=default_idx,
        help="This is the column the model will learn to predict."
    )
    
    # Validate target
    target_errors = validate_target(df, target_col)
    if target_errors:
        for err in target_errors:
            st.error(f"⚠️ {err}")
        st.stop()
    
    # Show data preview
    with st.expander("👀 Preview Dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    
    # ========================================================================
    # DATASET PROFILING
    # ========================================================================
    
    st.markdown('<div class="section-header"><h2>📊 Step 3: Dataset Profile</h2></div>', unsafe_allow_html=True)
    
    profile = profile_dataset(df, target_col)
    
    # Key metrics in columns
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Rows", f"{profile['n_rows']:,}")
    with m2:
        st.metric("Features", str(profile['n_features']))
    with m3:
        st.metric("Classes", str(profile['n_classes']))
    with m4:
        st.metric("Missing %", f"{profile['missing_ratio']}%")
    with m5:
        st.metric("Duplicates", str(profile['duplicate_rows']))
    
    prof_col1, prof_col2 = st.columns(2)
    
    with prof_col1:
        st.markdown("**📋 Dataset Overview**")
        overview_df = format_profile_for_display(profile)
        st.dataframe(overview_df, use_container_width=True, hide_index=True)
    
    with prof_col2:
        st.markdown("**📊 Class Distribution**")
        class_dist = profile['class_distribution']
        fig_class = px.bar(
            x=list(class_dist.keys()),
            y=list(class_dist.values()),
            labels={'x': 'Class', 'y': 'Count'},
            color=list(class_dist.values()),
            color_continuous_scale='Viridis'
        )
        fig_class.update_layout(
            showlegend=False, coloraxis_showscale=False,
            height=350, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_class, use_container_width=True)
    
    if profile.get('is_imbalanced'):
        st.warning(f"⚠️ Class imbalance detected (ratio: {profile['class_imbalance_ratio']}). "
                   "Results may be affected. Consider using F1 Score for evaluation.")
    
    # ========================================================================
    # ANALYZE BUTTON
    # ========================================================================
    
    st.markdown("---")
    
    analyze_button = st.button(
        "🚀 Run Algorithm Selection & Model Evaluation",
        type="primary",
        use_container_width=True
    )
    
    if analyze_button:
        # Store analysis state
        st.session_state['run_analysis'] = True
    
    if st.session_state.get('run_analysis', False):
        
        # ====================================================================
        # PREPROCESSING
        # ====================================================================
        
        with st.spinner("🔄 Preprocessing dataset..."):
            try:
                X, y, feature_names, target_le, prep_info = preprocess_dataset(df, target_col)
                st.success(f"✅ Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")
                st.stop()
        
        # ====================================================================
        # META-FEATURE EXTRACTION & ALGORITHM SELECTION
        # ====================================================================
        
        st.markdown('<div class="section-header"><h2>🎯 Step 4: Algorithm Selection (Meta-Learning)</h2></div>', unsafe_allow_html=True)
        
        with st.spinner("🔄 Extracting meta-features and running algorithm selection..."):
            try:
                meta_features = extract_meta_features(X, y)
                recommended_algo, confidence_info = get_meta_learning_recommendation(meta_features)
                
                algo_names = get_algorithm_name_mapping()
                display_name = algo_names.get(recommended_algo, recommended_algo)
                
            except Exception as e:
                st.error(f"❌ Algorithm selection failed: {str(e)}")
                recommended_algo = 'RandomForest'
                confidence_info = {'confidence': 0, 'meta_features_used': {}, 'neighbor_algorithms': []}
                display_name = 'Random Forest'
        
        # Show recommendation
        st.markdown(f"""
        <div class="recommendation-card">
            <h2>🏆 Recommended Algorithm</h2>
            <div class="algo-name">{display_name}</div>
            <div class="score">Meta-Learning Confidence: {confidence_info.get('confidence', 0):.1f}%</div>
            <p style="margin-top: 0.5rem; opacity: 0.8; font-size: 0.9rem;">
                Based on dataset meta-feature analysis and similarity to known dataset profiles
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Show meta-features
        with st.expander("🔬 Meta-Features Extracted", expanded=False):
            mf_display = confidence_info.get('meta_features_used', {})
            mf_df = pd.DataFrame([
                {"Meta-Feature": k, "Value": f"{v:.4f}"} 
                for k, v in mf_display.items()
            ])
            st.dataframe(mf_df, use_container_width=True, hide_index=True)
        
        # ====================================================================
        # MODEL EVALUATION
        # ====================================================================
        
        st.markdown('<div class="section-header"><h2>📈 Step 5: Model Comparison</h2></div>', unsafe_allow_html=True)
        
        with st.spinner("🔄 Training and evaluating all candidate models... This may take a moment."):
            try:
                results, comparison_df, best_model_name, split_data = evaluate_all_models(
                    X, y, recommended_algo=recommended_algo, test_size=test_size
                )
            except Exception as e:
                st.error(f"❌ Model evaluation failed: {str(e)}")
                st.stop()
        
        # Success message
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        st.success(f"✅ Evaluated {successful}/{len(results)} candidate algorithms successfully")
        
        # Comparison table
        st.markdown("**📊 Algorithm Performance Comparison**")
        
        # Highlight recommended and best
        display_df = comparison_df.copy()
        st.dataframe(
            display_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='#d1fae5')
                           .highlight_min(subset=['Training Time (s)'], color='#dbeafe'),
            use_container_width=True,
            hide_index=True
        )
        
        # Note about best algorithm
        if best_model_name:
            best_result = results[best_model_name]
            best_display = algo_names.get(best_model_name, best_model_name)
            
            if best_model_name == recommended_algo:
                st.info(f"✅ The meta-learning recommendation (**{best_display}**) achieved the highest "
                       f"F1 Score ({best_result['f1_score']:.4f}) among all evaluated candidates on this dataset.")
            else:
                rec_result = results.get(recommended_algo, {})
                rec_f1 = rec_result.get('f1_score', 0) if rec_result.get('status') == 'success' else 0
                st.info(
                    f"📊 **{best_display}** achieved the highest observed F1 Score ({best_result['f1_score']:.4f}) "
                    f"on this dataset. The meta-learning recommendation was **{algo_names.get(recommended_algo, recommended_algo)}** "
                    f"(F1: {rec_f1:.4f}). The actual best-performer may differ from the recommendation because "
                    f"meta-learning is based on similarity to known datasets, not direct evaluation."
                )
        
        # Performance charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Bar chart of F1 scores
            successful_df = comparison_df[comparison_df['Status'].str.contains('Success')].copy()
            if not successful_df.empty:
                colors = ['#667eea' if algo == recommended_algo else '#94a3b8' 
                         for algo in successful_df['Algorithm']]
                
                fig_f1 = px.bar(
                    successful_df,
                    x='Algorithm', y='F1 Score',
                    title='F1 Score by Algorithm',
                    color='Algorithm',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_f1.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_f1, use_container_width=True)
        
        with chart_col2:
            # Radar chart of metrics for top 5
            if not successful_df.empty:
                top5 = successful_df.head(5)
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                
                fig_radar = go.Figure()
                for _, row in top5.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[m] for m in metrics] + [row[metrics[0]]],
                        theta=metrics + [metrics[0]],
                        name=row['Algorithm'],
                        fill='toself',
                        opacity=0.6
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Top 5 — Metric Comparison",
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # Confusion matrix for best/recommended model
        with st.expander("📊 Confusion Matrix — Best Performing Model", expanded=False):
            if best_model_name and results[best_model_name]['status'] == 'success':
                cm = results[best_model_name]['confusion_matrix']
                classes = prep_info.get('target_classes', [str(i) for i in range(cm.shape[0])])
                
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[str(c) for c in classes],
                    y=[str(c) for c in classes],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig_cm.update_layout(
                    title=f"Confusion Matrix — {algo_names.get(best_model_name, best_model_name)}",
                    height=400
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # ====================================================================
        # EXPLAINABILITY (XAI)
        # ====================================================================
        
        st.markdown('<div class="section-header"><h2>🔍 Step 6: Explainability (XAI)</h2></div>', unsafe_allow_html=True)
        
        # Tab-based XAI sections
        xai_tab1, xai_tab2, xai_tab3 = st.tabs([
            "🏆 Why This Algorithm?", 
            "📊 Feature Importance", 
            "🔬 SHAP Analysis"
        ])
        
        # Use the best model for explanation
        explain_model_name = best_model_name if best_model_name else recommended_algo
        explain_result = results.get(explain_model_name, {})
        
        with xai_tab1:
            st.markdown("### Algorithm Selection Explanation")
            st.markdown(
                "This section explains why the meta-learning system recommended "
                "this particular algorithm. This is an **algorithm selection explanation**, "
                "distinct from model prediction explanations."
            )
            
            explanation_text = explain_algorithm_selection(
                meta_features, confidence_info, 
                algo_names.get(recommended_algo, recommended_algo)
            )
            st.markdown(explanation_text)
            
            # Meta-learning score distribution
            if confidence_info.get('all_scores'):
                st.markdown("#### Meta-Learning Score Distribution")
                scores = confidence_info['all_scores']
                fig_scores = px.bar(
                    x=list(scores.keys()),
                    y=list(scores.values()),
                    labels={'x': 'Algorithm', 'y': 'Similarity Score (%)'},
                    color=list(scores.values()),
                    color_continuous_scale='Viridis'
                )
                fig_scores.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    height=350
                )
                st.plotly_chart(fig_scores, use_container_width=True)
        
        with xai_tab2:
            st.markdown("### Feature Importance — Model Prediction Explanation")
            st.markdown(
                "This shows which features are most important for the trained model's predictions. "
                "This is a **model prediction explanation**, not an algorithm selection explanation."
            )
            
            if explain_result.get('status') == 'success':
                model = explain_result['trained_model']
                importance = get_feature_importance(model, feature_names, explain_model_name)
                
                if importance:
                    fig_imp = plot_feature_importance(
                        importance,
                        title=f"Feature Importance — {algo_names.get(explain_model_name, explain_model_name)}"
                    )
                    if fig_imp:
                        st.pyplot(fig_imp)
                        plt.close(fig_imp)
                    
                    # Also show as table
                    with st.expander("📋 Feature Importance Table"):
                        imp_df = pd.DataFrame([
                            {"Feature": k, "Importance": f"{v:.6f}"} 
                            for k, v in importance.items()
                        ])
                        st.dataframe(imp_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Feature importance is not directly available for this model type. See SHAP analysis.")
            else:
                st.warning("Model training was not successful. Cannot compute feature importance.")
        
        with xai_tab3:
            st.markdown("### SHAP Analysis — Model Prediction Explanation")
            st.markdown(
                "SHAP (SHapley Additive exPlanations) provides unified feature importance "
                "based on game-theoretic principles. Each feature's contribution to individual "
                "predictions is quantified."
            )
            
            if explain_result.get('status') == 'success':
                model = explain_result['trained_model']
                X_train = split_data['X_train']
                X_test = split_data['X_test']
                
                with st.spinner("🔄 Computing SHAP values... This may take a moment."):
                    try:
                        explainer, explainer_type = get_shap_explainer(model, X_train, explain_model_name)
                        
                        if explainer is not None and not isinstance(explainer_type, str):
                            shap_values, X_sample = compute_shap_values(
                                explainer, X_test, explainer_type, explain_model_name
                            )
                            
                            if shap_values is not None:
                                st.success(f"✅ SHAP values computed using **{explainer_type.capitalize()}Explainer**")
                                
                                # SHAP summary bar plot
                                st.markdown("#### SHAP Feature Importance (Mean |SHAP|)")
                                fig_shap = plot_shap_summary(
                                    shap_values, X_sample, feature_names, 
                                    algo_names.get(explain_model_name, explain_model_name)
                                )
                                if fig_shap:
                                    st.pyplot(fig_shap)
                                    plt.close(fig_shap)
                                
                                # SHAP beeswarm plot
                                st.markdown("#### SHAP Beeswarm Plot")
                                try:
                                    fig_bee = plot_shap_beeswarm(shap_values, X_sample, feature_names)
                                    if fig_bee:
                                        st.pyplot(fig_bee)
                                        plt.close(fig_bee)
                                except Exception:
                                    st.info("Beeswarm plot could not be generated for this model type.")
                            else:
                                st.warning("SHAP values could not be computed. Showing feature importance instead.")
                                importance = get_feature_importance(model, feature_names, explain_model_name)
                                if importance:
                                    fig_imp = plot_feature_importance(importance, title="Feature Importance (Fallback)")
                                    if fig_imp:
                                        st.pyplot(fig_imp)
                                        plt.close(fig_imp)
                        
                        elif isinstance(explainer_type, str) and 'error' in explainer_type:
                            st.warning(f"SHAP explainer could not be created: {explainer_type}")
                            importance = get_feature_importance(model, feature_names, explain_model_name)
                            if importance:
                                fig_imp = plot_feature_importance(importance, title="Feature Importance (Fallback)")
                                if fig_imp:
                                    st.pyplot(fig_imp)
                                    plt.close(fig_imp)
                    
                    except Exception as e:
                        st.warning(f"SHAP analysis encountered an error: {str(e)}")
                        st.info("Falling back to sklearn feature importance.")
                        importance = get_feature_importance(model, feature_names, explain_model_name)
                        if importance:
                            fig_imp = plot_feature_importance(importance, title="Feature Importance (Fallback)")
                            if fig_imp:
                                st.pyplot(fig_imp)
                                plt.close(fig_imp)
            else:
                st.warning("Model training was not successful. Cannot perform SHAP analysis.")
        
        # ====================================================================
        # RESULTS SUMMARY
        # ====================================================================
        
        st.markdown('<div class="section-header"><h2>📋 Step 7: Results Summary</h2></div>', unsafe_allow_html=True)
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("#### 📊 Analysis Summary")
            st.markdown(f"""
            | Aspect | Detail |
            |--------|--------|
            | **Dataset** | {profile['n_rows']:,} rows × {profile['n_cols']} columns |
            | **Target** | `{target_col}` ({profile['n_classes']} classes) |
            | **Features Used** | {len(feature_names)} |
            | **Test Split** | {test_size*100:.0f}% |
            | **Models Evaluated** | {successful}/{len(results)} |
            | **Meta-Learning Recommendation** | {algo_names.get(recommended_algo, recommended_algo)} |
            | **Best Observed Performance** | {algo_names.get(best_model_name, best_model_name) if best_model_name else 'N/A'} |
            """)
        
        with summary_col2:
            st.markdown("#### 🏆 Best Model Metrics")
            if best_model_name and results[best_model_name]['status'] == 'success':
                br = results[best_model_name]
                st.markdown(f"""
                | Metric | Score |
                |--------|-------|
                | **Algorithm** | {algo_names.get(best_model_name, best_model_name)} |
                | **Accuracy** | {br['accuracy']:.4f} |
                | **Precision** | {br['precision']:.4f} |
                | **Recall** | {br['recall']:.4f} |
                | **F1 Score** | {br['f1_score']:.4f} |
                | **Training Time** | {br['training_time']:.3f}s |
                """)
        
        # Downloadable results
        st.markdown("---")
        st.markdown("#### 💾 Download Results")
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            csv_results = comparison_df.to_csv(index=False)
            st.download_button(
                "📥 Download Model Comparison (CSV)",
                csv_results,
                "suggestalgo_model_comparison.csv",
                "text/csv"
            )
        
        with dl_col2:
            # Meta-features CSV
            mf_df = pd.DataFrame([meta_features])
            mf_csv = mf_df.to_csv(index=False)
            st.download_button(
                "📥 Download Meta-Features (CSV)",
                mf_csv,
                "suggestalgo_meta_features.csv",
                "text/csv"
            )

else:
    # No data loaded state
    st.info("👆 Upload a CSV dataset or select a sample dataset to begin analysis.")
    
    st.markdown("### 🎯 What this application does:")
    st.markdown("""
    1. **Upload** your classification dataset (CSV format)
    2. **Select** the target column for prediction
    3. **Profile** your dataset automatically
    4. **Recommend** the best ML algorithm using meta-learning
    5. **Evaluate** multiple candidate algorithms
    6. **Explain** why the recommendation was made using XAI/SHAP
    """)
    
    st.markdown("### 📚 Supported Algorithms:")
    algo_cols = st.columns(3)
    algos = [
        ("Random Forest", "Ensemble of decision trees"),
        ("Decision Tree", "Simple tree-based classifier"),
        ("Logistic Regression", "Linear classification model"),
        ("SVC", "Support Vector Classifier"),
        ("Extra Trees", "Extremely randomized trees"),
        ("Gradient Boosting", "Sequential tree boosting"),
        ("AdaBoost", "Adaptive boosting"),
        ("XGBoost", "Extreme gradient boosting"),
        ("SGD Classifier", "Stochastic gradient descent"),
    ]
    for i, (name, desc) in enumerate(algos):
        with algo_cols[i % 3]:
            st.markdown(f"**{name}**  \n_{desc}_")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <strong>SuggestAlgo AI</strong> — Explainable AI for Machine Learning Algorithm Selection<br>
    Foundation: <a href="https://github.com/LeMGarouani/AMLBID" target="_blank">AMLBID</a> by LeMGarouani et al.<br>
    Built with Streamlit • Scikit-learn • SHAP • Plotly
</div>
""", unsafe_allow_html=True)
