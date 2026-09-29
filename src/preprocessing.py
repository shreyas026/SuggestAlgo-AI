"""
SuggestAlgo AI - Preprocessing Module

Handles data preprocessing for ML model training.
Supports both classification and regression tasks.
Adapted from AMLBID's loader.py preprocessing functions (numeric_impute, cat_data_encode).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_dataset(df, target_col, problem_type=None):
    """
    Preprocess dataset for ML model training.

    Adapted from AMLBID's preprocessing pipeline:
    - AMLBID's numeric_impute() → our numerical imputation
    - AMLBID's cat_data_encode() → our categorical encoding
    - Added: constant column removal, scaling

    For regression: target is kept as float (not LabelEncoded).
    For classification: target is LabelEncoded.

    Args:
        df: Input DataFrame
        target_col: Name of the target column
        problem_type: 'classification', 'regression', or None (auto-detect)

    Returns:
        tuple: (X_processed, y_encoded, feature_names, label_encoder_or_None, preprocessing_info)
    """
    from src.model_evaluation import detect_problem_type

    preprocessing_info = {}

    # Separate features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Remove constant columns (as AMLBID does)
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        preprocessing_info['removed_constant_cols'] = constant_cols

    # Remove ID-like columns (as AMLBID does - checks for 'id' column)
    id_cols = [col for col in X.columns if col.lower() in ['id', 'index', 'row_id', 'row_number']]
    if id_cols:
        X = X.drop(columns=id_cols)
        preprocessing_info['removed_id_cols'] = id_cols

    # Identify column types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    preprocessing_info['numerical_cols'] = numerical_cols
    preprocessing_info['categorical_cols'] = categorical_cols

    # Handle numerical missing values (AMLBID uses mean imputation by default)
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='mean')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

    # Handle categorical columns
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

        cat_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            cat_encoders[col] = le
        preprocessing_info['cat_encoders'] = cat_encoders

    # Handle missing values in target
    y = y.dropna()
    valid_indices = y.index
    X = X.loc[valid_indices]

    # Detect problem type if not given
    if problem_type is None:
        problem_type = detect_problem_type(y.values if hasattr(y, 'values') else y)

    preprocessing_info['problem_type'] = problem_type

    target_le = None

    if problem_type == 'regression':
        # Keep target as float for regression
        try:
            y_encoded = pd.to_numeric(y, errors='coerce').fillna(0).values.astype(float)
        except Exception:
            y_encoded = y.values.astype(float)
        preprocessing_info['target_classes'] = None
        preprocessing_info['n_classes'] = None
    else:
        # Encode target as class labels for classification
        target_le = LabelEncoder()
        y_encoded = target_le.fit_transform(y.astype(str))
        preprocessing_info['target_classes'] = target_le.classes_.tolist()
        preprocessing_info['n_classes'] = len(target_le.classes_)

    # Scale numerical features
    scaler = StandardScaler()
    if numerical_cols:
        remaining_num_cols = [c for c in numerical_cols if c in X.columns]
        if remaining_num_cols:
            X[remaining_num_cols] = scaler.fit_transform(X[remaining_num_cols])

    feature_names = X.columns.tolist()

    return X, y_encoded, feature_names, target_le, preprocessing_info
