"""
SuggestAlgo AI - Preprocessing Module

Handles data preprocessing for ML model training.
Adapted from AMLBID's loader.py preprocessing functions (numeric_impute, cat_data_encode).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_dataset(df, target_col):
    """
    Preprocess dataset for ML model training.
    
    Adapted from AMLBID's preprocessing pipeline:
    - AMLBID's numeric_impute() → our numerical imputation
    - AMLBID's cat_data_encode() → our categorical encoding
    - Added: constant column removal, scaling
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        tuple: (X_processed, y_encoded, feature_names, label_encoder, preprocessing_info)
    """
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
        # Fill missing categorical values with mode
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        
        # Label encode categorical columns (as AMLBID does with LabelEncoder)
        cat_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            cat_encoders[col] = le
        preprocessing_info['cat_encoders'] = cat_encoders
    
    # Encode target variable
    target_le = LabelEncoder()
    
    # Handle missing values in target
    y = y.dropna()
    valid_indices = y.index
    X = X.loc[valid_indices]
    
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
