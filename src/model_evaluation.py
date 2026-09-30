"""
SuggestAlgo AI - Model Evaluation Module

Trains and evaluates candidate ML algorithms on the user's dataset.
Supports BOTH classification and regression tasks.
Uses the same algorithm pool as AMLBID (see AMLBID/loader.py imports).
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier,
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib

warnings.filterwarnings('ignore')

# Random state for reproducibility
RANDOM_STATE = 42


def detect_problem_type(y):
    """
    Detect whether the target variable represents a classification or regression task.

    Heuristic (same as AMLBID methodology):
    - If categorical, object, bool, or string dtype → Classification
    - If integer with ≤20 unique values → Classification
    - If float or many unique integer values → Regression
    - Returns 'classification' or 'regression'
    """
    # Convert to numpy array to handle pandas Series, Categorical, etc.
    if hasattr(y, 'values'):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)

    n_total = len(y_arr)

    # Handle object / string / categorical dtype → always classification
    try:
        # Try to convert to float; if it fails, it's categorical/string
        y_numeric = y_arr.astype(float)
    except (ValueError, TypeError):
        return 'classification'

    n_unique = len(np.unique(y_numeric))

    # Check original dtype for integer signals
    try:
        original_dtype = np.array(y_arr).dtype
        if original_dtype == bool:
            return 'classification'
        if np.issubdtype(original_dtype, np.integer):
            return 'classification' if n_unique <= 20 else 'regression'
        if np.issubdtype(original_dtype, np.floating):
            if n_unique / max(n_total, 1) > 0.05 or n_unique > 20:
                return 'regression'
            else:
                return 'classification'
    except (TypeError, AttributeError):
        pass

    # Fallback based on unique value count
    if n_unique <= 20:
        return 'classification'
    return 'regression'


def get_candidate_classifiers(n_classes=2, n_samples=100, n_features=10):
    """
    Get candidate classification models.
    
    These are the same algorithms used in AMLBID (from AMLBID/loader.py):
    - RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    - LogisticRegression, SGDClassifier
    - DecisionTreeClassifier
    - SVC
    - XGBClassifier
    """
    models = {}
    max_iter = 1000 if n_samples < 5000 else 500

    models['RandomForest'] = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    models['DecisionTree'] = DecisionTreeClassifier(
        max_depth=None, random_state=RANDOM_STATE
    )
    models['LogisticRegression'] = LogisticRegression(
        max_iter=max_iter, random_state=RANDOM_STATE, solver='lbfgs'
    )
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
        n_estimators=100, random_state=RANDOM_STATE,
        eval_metric='logloss', verbosity=0
    )
    models['SGDClassifier'] = SGDClassifier(
        max_iter=max_iter, random_state=RANDOM_STATE, loss='modified_huber'
    )
    return models


def get_candidate_regressors(n_samples=100, n_features=10):
    """
    Get candidate regression models.

    Pool:
    - LinearRegression, Ridge, Lasso, ElasticNet
    - DecisionTreeRegressor
    - RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    - SVR
    - XGBRegressor
    """
    models = {}

    models['LinearRegression'] = LinearRegression()
    models['Ridge'] = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    models['Lasso'] = Lasso(alpha=0.1, max_iter=2000, random_state=RANDOM_STATE)
    models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=RANDOM_STATE)
    models['DecisionTree'] = DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE)
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    models['ExtraTrees'] = ExtraTreesRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=100, random_state=RANDOM_STATE, max_depth=3
    )
    # SVR is slow on large datasets — skip if too large
    if n_samples <= 5000:
        models['SVR'] = SVR(kernel='rbf', C=1.0)
    models['XGBoost'] = XGBRegressor(
        n_estimators=100, random_state=RANDOM_STATE, verbosity=0
    )

    return models


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_single_classifier(model, X_train, X_test, y_train, y_test,
                                model_name, n_classes, X_full=None, y_full=None):
    """Train and evaluate a single classifier with 5-Fold Stratified CV."""
    result = {'algorithm': model_name, 'task': 'classification', 'status': 'success', 'error': None}

    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)
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

        # 5-Fold Stratified Cross Validation
        if X_full is not None and y_full is not None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(model, X_full, y_full, cv=skf, scoring='accuracy')
            result['cv_accuracy_mean'] = round(float(np.mean(cv_scores)), 4)
            result['cv_accuracy_std'] = round(float(np.std(cv_scores)), 4)
        else:
            result['cv_accuracy_mean'] = result['accuracy']
            result['cv_accuracy_std'] = 0.0

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        for k in ['accuracy', 'precision', 'recall', 'f1_score', 'training_time',
                  'cv_accuracy_mean', 'cv_accuracy_std']:
            result[k] = 0

    return result


# ─────────────────────────────────────────────────────────────
# REGRESSION EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_single_regressor(model, X_train, X_test, y_train, y_test,
                               model_name, X_full=None, y_full=None):
    """Train and evaluate a single regressor with 5-Fold KFold CV."""
    result = {'algorithm': model_name, 'task': 'regression', 'status': 'success', 'error': None}

    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)

        result['mae'] = round(mean_absolute_error(y_test, y_pred), 4)
        result['mse'] = round(mean_squared_error(y_test, y_pred), 4)
        result['rmse'] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        result['r2'] = round(r2_score(y_test, y_pred), 4)
        result['training_time'] = round(train_time, 3)
        result['trained_model'] = model
        result['predictions'] = y_pred

        # 5-Fold KFold Cross Validation (R² score)
        if X_full is not None and y_full is not None:
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(model, X_full, y_full, cv=kf, scoring='r2')
            result['cv_r2_mean'] = round(float(np.mean(cv_scores)), 4)
            result['cv_r2_std'] = round(float(np.std(cv_scores)), 4)
        else:
            result['cv_r2_mean'] = result['r2']
            result['cv_r2_std'] = 0.0

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        for k in ['mae', 'mse', 'rmse', 'r2', 'training_time', 'cv_r2_mean', 'cv_r2_std']:
            result[k] = 0

    return result


# ─────────────────────────────────────────────────────────────
# UNIFIED EVALUATION ENTRY POINT
# ─────────────────────────────────────────────────────────────

# Legacy alias kept for backward compatibility (used across app.py and tests)
def get_candidate_models(n_classes=2, n_samples=100, n_features=10):
    """Backward-compatible alias for get_candidate_classifiers."""
    return get_candidate_classifiers(n_classes, n_samples, n_features)


def evaluate_single_model(model, X_train, X_test, y_train, y_test,
                           model_name, n_classes, X_full=None, y_full=None):
    """Backward-compatible alias for evaluate_single_classifier."""
    return evaluate_single_classifier(
        model, X_train, X_test, y_train, y_test, model_name, n_classes, X_full, y_full
    )


def evaluate_all_models(X, y, recommended_algo=None, test_size=0.3, problem_type=None):
    """
    Evaluate all candidate models on the dataset with Cross Validation.

    Auto-detects problem type (classification vs regression) unless explicitly provided.
    Returns results dict, comparison DataFrame, best model name, and split data.
    """
    # Auto-detect problem type if not provided
    if problem_type is None:
        problem_type = detect_problem_type(y)

    n_samples, n_features = X.shape

    if problem_type == 'regression':
        return _evaluate_regression(X, y, n_samples, n_features, test_size)
    else:
        return _evaluate_classification(X, y, n_samples, n_features, test_size)


def _evaluate_classification(X, y, n_samples, n_features, test_size):
    """Internal: run classification benchmarking."""
    n_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    models = get_candidate_classifiers(n_classes, n_samples, n_features)
    results = {}

    for name, model in models.items():
        result = evaluate_single_classifier(
            model, X_train, X_test, y_train, y_test, name, n_classes, X_full=X, y_full=y
        )
        results[name] = result

    comparison_data = []
    for name, result in results.items():
        if result['status'] == 'success':
            comparison_data.append({
                'Algorithm': name,
                'Accuracy': result['accuracy'],
                'CV Accuracy Mean': result['cv_accuracy_mean'],
                'CV Accuracy Std': result['cv_accuracy_std'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1_score'],
                'Training Time (s)': result['training_time'],
                'Status': '✅ Success'
            })
        else:
            comparison_data.append({
                'Algorithm': name,
                'Accuracy': 0, 'CV Accuracy Mean': 0, 'CV Accuracy Std': 0,
                'Precision': 0, 'Recall': 0, 'F1 Score': 0,
                'Training Time (s)': 0,
                'Status': f'❌ {result["error"][:50]}'
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

    successful = {k: v for k, v in results.items() if v['status'] == 'success'}
    best_model_name = max(successful, key=lambda k: successful[k]['f1_score']) if successful else None

    split_data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    return results, comparison_df, best_model_name, split_data


def _evaluate_regression(X, y, n_samples, n_features, test_size):
    """Internal: run regression benchmarking."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    models = get_candidate_regressors(n_samples, n_features)
    results = {}

    for name, model in models.items():
        result = evaluate_single_regressor(
            model, X_train, X_test, y_train, y_test, name, X_full=X, y_full=y
        )
        results[name] = result

    comparison_data = []
    for name, result in results.items():
        if result['status'] == 'success':
            comparison_data.append({
                'Algorithm': name,
                'MAE': result['mae'],
                'MSE': result['mse'],
                'RMSE': result['rmse'],
                'R² Score': result['r2'],
                'CV R² Mean': result['cv_r2_mean'],
                'CV R² Std': result['cv_r2_std'],
                'Training Time (s)': result['training_time'],
                'Status': '✅ Success'
            })
        else:
            comparison_data.append({
                'Algorithm': name,
                'MAE': 0, 'MSE': 0, 'RMSE': 0,
                'R² Score': 0, 'CV R² Mean': 0, 'CV R² Std': 0,
                'Training Time (s)': 0,
                'Status': f'❌ {result["error"][:50]}'
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('R² Score', ascending=False).reset_index(drop=True)

    successful = {k: v for k, v in results.items() if v['status'] == 'success'}
    best_model_name = max(successful, key=lambda k: successful[k]['r2']) if successful else None

    split_data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    return results, comparison_df, best_model_name, split_data


# ─────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────

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
