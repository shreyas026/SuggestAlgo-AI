"""
SuggestAlgo AI - Explainability Module

Provides XAI explanations using SHAP and feature importance.

AMLBID has its own Explainer module (AMLBID/Explainer/) built on Dash with SHAP.
We adapt the explainability concepts for Streamlit, using:
1. SHAP TreeExplainer for tree-based models
2. SHAP LinearExplainer for linear models
3. SHAP KernelExplainer as fallback (with sampling for performance)
4. Sklearn feature_importances_ where available

We clearly distinguish:
- Model prediction explanation (SHAP on the trained model)
- Algorithm selection explanation (meta-feature importance for recommendation)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
import io

warnings.filterwarnings('ignore')

# Maximum rows for SHAP computation to keep it fast
MAX_SHAP_ROWS = 500


def get_shap_explainer(model, X_train, model_name):
    """
    Create the appropriate SHAP explainer based on model type.
    
    Mirrors AMLBID's approach of using TreeExplainer for tree models
    and KernelExplainer for others. Fallback to KernelExplainer if
    TreeExplainer fails (e.g., multiclass GradientBoosting).
    """
    try:
        # Tree-based models → TreeExplainer (fast)
        if model_name in ['RandomForest', 'DecisionTree', 'ExtraTrees', 
                          'GradientBoosting', 'XGBoost']:
            try:
                explainer = shap.TreeExplainer(model)
                return explainer, 'tree'
            except Exception:
                # Fallback for multiclass GradientBoosting or unsupported tree variants
                if len(X_train) > 100:
                    X_bg = shap.sample(X_train, 100)
                else:
                    X_bg = X_train
                explainer = shap.KernelExplainer(model.predict_proba, X_bg)
                return explainer, 'kernel'
        
        # Linear models → LinearExplainer
        elif model_name in ['LogisticRegression', 'SGDClassifier']:
            # Sample training data for background
            if len(X_train) > MAX_SHAP_ROWS:
                X_bg = X_train.sample(n=MAX_SHAP_ROWS, random_state=42)
            else:
                X_bg = X_train
            explainer = shap.LinearExplainer(model, X_bg)
            return explainer, 'linear'
        
        # Other models → KernelExplainer (slow, use small sample)
        else:
            if len(X_train) > 100:
                X_bg = shap.sample(X_train, 100)
            else:
                X_bg = X_train
            explainer = shap.KernelExplainer(model.predict_proba, X_bg)
            return explainer, 'kernel'
    
    except Exception as e:
        return None, f'error: {str(e)}'


def compute_shap_values(explainer, X_test, explainer_type, model_name):
    """Compute SHAP values with appropriate sampling."""
    try:
        # Sample test data if too large
        if len(X_test) > MAX_SHAP_ROWS:
            X_sample = X_test.sample(n=MAX_SHAP_ROWS, random_state=42)
        else:
            X_sample = X_test.copy()
        
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, X_sample
    
    except Exception as e:
        return None, None


def get_feature_importance(model, feature_names, model_name):
    """
    Get feature importance from sklearn models.
    Works for tree-based models and linear models.
    """
    importance_dict = {}
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
            importance_dict = dict(zip(feature_names, importances))
        else:
            return None
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return importance_dict
    
    except Exception:
        return None


def plot_feature_importance(importance_dict, title="Feature Importance", top_n=15):
    """Create a feature importance bar chart."""
    if importance_dict is None or len(importance_dict) == 0:
        return None
    
    # Take top N features
    items = list(importance_dict.items())[:top_n]
    features = [item[0] for item in items]
    values = [item[1] for item in items]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))[::-1]
    
    bars = ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_shap_summary(shap_values, X_sample, feature_names, model_name):
    """Create SHAP summary plot."""
    try:
        if shap_values is None:
            return None
            
        fig, ax = plt.subplots(figsize=(10, max(4, min(len(feature_names), 15) * 0.4)))
        
        # Handle multiclass/multi-dimensional SHAP values robustly
        if isinstance(shap_values, list):
            abs_vals = np.abs(np.array(shap_values)) # (classes, samples, features)
            if abs_vals.ndim == 3 and abs_vals.shape[2] == len(feature_names):
                feature_importance = np.mean(abs_vals, axis=(0, 1))
            else:
                feature_importance = np.mean(abs_vals, axis=(0, 2))
        else:
            shap_arr = np.array(shap_values)
            abs_vals = np.abs(shap_arr)
            if abs_vals.ndim == 3:
                if abs_vals.shape[1] == len(feature_names):
                    feature_importance = np.mean(abs_vals, axis=(0, 2))
                else:
                    feature_importance = np.mean(abs_vals, axis=(0, 1))
            elif abs_vals.ndim == 2:
                feature_importance = np.mean(abs_vals, axis=0)
            else:
                feature_importance = np.ravel(abs_vals)[:len(feature_names)]
        
        # Ensure 1D array matching feature_names length
        feature_importance = np.ravel(feature_importance)[:len(feature_names)]
        
        # Sort by importance
        top_n = min(15, len(feature_names))
        sorted_idx = np.argsort(feature_importance)[-top_n:]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        
        ax.barh(range(top_n), feature_importance[sorted_idx], color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title(f'SHAP Feature Importance — {model_name}', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"plot_shap_summary exception: {e}")
        return None


def plot_shap_beeswarm(shap_values, X_sample, feature_names):
    """Create SHAP beeswarm plot using matplotlib."""
    try:
        fig = plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            # Binary classification: use class 1
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        
        shap.summary_plot(sv, X_sample, feature_names=feature_names, 
                         show=False, max_display=15)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def explain_algorithm_selection(meta_features, confidence_info, recommended_algo):
    """
    Generate explanation for why the meta-learning system recommended this algorithm.
    
    This is the algorithm SELECTION explanation (not model prediction explanation).
    """
    explanation = []
    
    explanation.append(f"### Why {recommended_algo} was recommended\n")
    explanation.append(
        f"The meta-learning system analyzed your dataset's characteristics (meta-features) "
        f"and found it most similar to datasets where **{recommended_algo}** performed best.\n"
    )
    
    # Key meta-features that influenced the decision
    explanation.append("#### Key Dataset Characteristics Considered:\n")
    mf = confidence_info.get('meta_features_used', {})
    
    n_instances = mf.get('nr_instances', 0)
    n_features = mf.get('nr_features', 0)
    n_classes = mf.get('nr_classes', 0)
    
    if n_instances < 500:
        explanation.append(f"- **Small dataset** ({int(n_instances)} instances): Simpler models often generalize better")
    elif n_instances < 5000:
        explanation.append(f"- **Medium dataset** ({int(n_instances)} instances): Suitable for most ensemble methods")
    else:
        explanation.append(f"- **Large dataset** ({int(n_instances)} instances): Ensemble methods and boosting tend to excel")
    
    if n_features < 10:
        explanation.append(f"- **Low dimensionality** ({int(n_features)} features): Linear models and SVMs are competitive")
    elif n_features < 50:
        explanation.append(f"- **Moderate dimensionality** ({int(n_features)} features): Tree ensembles typically perform well")
    else:
        explanation.append(f"- **High dimensionality** ({int(n_features)} features): Regularized models or ensemble methods preferred")
    
    if n_classes == 2:
        explanation.append("- **Binary classification**: All candidate algorithms are applicable")
    else:
        explanation.append(f"- **Multiclass ({int(n_classes)} classes)**: Tree-based ensembles often handle multiclass natively")
    
    imbalance = mf.get('class_imbalance', 1)
    if imbalance > 3:
        explanation.append(f"- **Imbalanced classes** (ratio: {imbalance:.1f}): Ensemble methods can handle class imbalance")
    
    explanation.append(f"\n#### Meta-Learning Confidence: {confidence_info.get('confidence', 0):.1f}%\n")
    
    explanation.append("#### Nearest Neighbor Algorithms (from knowledge base):\n")
    for algo in confidence_info.get('neighbor_algorithms', []):
        explanation.append(f"- {algo}")
    
    explanation.append(
        "\n> **Note**: This recommendation is based on meta-learning — finding similar datasets "
        "in a knowledge base and recommending the algorithm that performed best on those similar "
        "datasets. The actual performance on your specific dataset is verified by training and "
        "evaluating the model (see Model Comparison section)."
    )
    
    return "\n".join(explanation)
