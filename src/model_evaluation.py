"""
SuggestAlgo AI - Model Evaluation Module

Trains and evaluates candidate ML algorithms on the user's dataset.
Uses the same algorithm pool as AMLBID (see AMLBID/loader.py imports).
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings('ignore')

# Random state for reproducibility
RANDOM_STATE = 42


def get_candidate_models(n_classes=2, n_samples=100, n_features=10):
    """
    Get candidate ML models for evaluation.
    
    These are the same algorithms used in AMLBID (from AMLBID/loader.py):
    - RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    - LogisticRegression, SGDClassifier
    - DecisionTreeClassifier
    - SVC
    - XGBClassifier
    
    Hyperparameters are set to reasonable defaults with adjustments for dataset size.
    """
    models = {}
    
    # Adjust max_iter for convergence on larger datasets
    max_iter = 1000 if n_samples < 5000 else 500
    
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    
    models['DecisionTree'] = DecisionTreeClassifier(
        max_depth=None, random_state=RANDOM_STATE
    )
    
    models['LogisticRegression'] = LogisticRegression(
        max_iter=max_iter, random_state=RANDOM_STATE, multi_class='auto', solver='lbfgs'
    )
    
    # SVC with probability estimates for SHAP compatibility
    # Use linear kernel for large datasets to keep training time reasonable
    if n_samples > 5000:
        models['SVC'] = SVC(
            kernel='linear', probability=True, random_state=RANDOM_STATE, max_iter=max_iter
        )
    else:
        models['SVC'] = SVC(
            kernel='rbf', probability=True, random_state=RANDOM_STATE
        )
    
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE, max_depth=3
    )
    
    models['AdaBoost'] = AdaBoostClassifier(
        n_estimators=100, random_state=RANDOM_STATE, algorithm='SAMME'
    )
    
    models['XGBoost'] = XGBClassifier(
        n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric='logloss', verbosity=0
    )
    
    models['SGDClassifier'] = SGDClassifier(
        max_iter=max_iter, random_state=RANDOM_STATE, loss='modified_huber'  # enables predict_proba
    )
    
    return models


def evaluate_single_model(model, X_train, X_test, y_train, y_test, model_name, n_classes):
    """Train and evaluate a single model with error handling."""
    result = {
        'algorithm': model_name,
        'status': 'success',
        'error': None
    }
    
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        avg = 'weighted' if n_classes > 2 else 'binary'
        
        result['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
        result['precision'] = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
        result['recall'] = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
        result['f1_score'] = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)
        result['training_time'] = round(train_time, 3)
        result['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        result['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        result['trained_model'] = model
        result['predictions'] = y_pred
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['accuracy'] = 0
        result['precision'] = 0
        result['recall'] = 0
        result['f1_score'] = 0
        result['training_time'] = 0
    
    return result


def evaluate_all_models(X, y, recommended_algo=None, test_size=0.3):
    """
    Evaluate all candidate models on the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        recommended_algo: The algorithm recommended by meta-learning
        test_size: Test set proportion
        
    Returns:
        tuple: (results_dict, comparison_df, best_model_name)
    """
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape
    
    # Stratified train/test split for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    models = get_candidate_models(n_classes, n_samples, n_features)
    results = {}
    
    for name, model in models.items():
        result = evaluate_single_model(
            model, X_train, X_test, y_train, y_test, name, n_classes
        )
        results[name] = result
    
    # Build comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        if result['status'] == 'success':
            comparison_data.append({
                'Algorithm': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1_score'],
                'Training Time (s)': result['training_time'],
                'Status': '✅ Success'
            })
        else:
            comparison_data.append({
                'Algorithm': name,
                'Accuracy': 0,
                'Precision': 0,
                'Recall': 0,
                'F1 Score': 0,
                'Training Time (s)': 0,
                'Status': f'❌ {result["error"][:50]}'
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    # Find best performing model
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    if successful_results:
        best_model_name = max(successful_results, key=lambda k: successful_results[k]['f1_score'])
    else:
        best_model_name = None
    
    # Store split data for SHAP
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    return results, comparison_df, best_model_name, split_data


def save_model(model, filepath):
    """Save trained model to disk."""
    try:
        joblib.dump(model, filepath)
        return True
    except Exception:
        return False


def load_model(filepath):
    """Load trained model from disk."""
    try:
        return joblib.load(filepath)
    except Exception:
        return None
