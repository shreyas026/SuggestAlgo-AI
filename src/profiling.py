"""
SuggestAlgo AI - Dataset Profiling Module

Generates comprehensive dataset profiles for display and meta-feature extraction.
"""

import pandas as pd
import numpy as np


def profile_dataset(df, target_col):
    """
    Generate a comprehensive dataset profile.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        dict: Profile information
    """
    profile = {}
    
    # Basic dimensions
    profile['n_rows'] = df.shape[0]
    profile['n_cols'] = df.shape[1]
    profile['n_features'] = df.shape[1] - 1  # Exclude target
    
    # Target info
    profile['target_col'] = target_col
    profile['n_classes'] = df[target_col].nunique()
    profile['class_distribution'] = df[target_col].value_counts().to_dict()
    
    # Check class imbalance
    class_counts = df[target_col].value_counts()
    if len(class_counts) >= 2:
        imbalance_ratio = class_counts.max() / class_counts.min()
        profile['class_imbalance_ratio'] = round(imbalance_ratio, 2)
        profile['is_imbalanced'] = imbalance_ratio > 3.0
    else:
        profile['class_imbalance_ratio'] = 1.0
        profile['is_imbalanced'] = False
    
    # Column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Remove target from feature counts
    if target_col in numerical_cols:
        numerical_feature_cols = [c for c in numerical_cols if c != target_col]
    else:
        numerical_feature_cols = numerical_cols
    
    if target_col in categorical_cols:
        categorical_feature_cols = [c for c in categorical_cols if c != target_col]
    else:
        categorical_feature_cols = categorical_cols
    
    profile['numerical_features'] = len(numerical_feature_cols)
    profile['categorical_features'] = len(categorical_feature_cols)
    profile['numerical_cols'] = numerical_feature_cols
    profile['categorical_cols'] = categorical_feature_cols
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    profile['total_missing'] = int(total_missing)
    profile['missing_ratio'] = round(total_missing / df.size * 100, 2)
    profile['cols_with_missing'] = int((df.isnull().sum() > 0).sum())
    
    # Per-column missing values
    missing_per_col = df.isnull().sum()
    profile['missing_per_col'] = missing_per_col[missing_per_col > 0].to_dict()
    
    # Duplicate rows
    profile['duplicate_rows'] = int(df.duplicated().sum())
    
    # Numerical statistics
    if numerical_feature_cols:
        num_stats = df[numerical_feature_cols].describe().round(3)
        profile['numerical_stats'] = num_stats
    
    # Unique values per column
    profile['unique_values'] = {col: int(df[col].nunique()) for col in df.columns}
    
    # Data types
    profile['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Memory usage
    profile['memory_mb'] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    
    return profile


def format_profile_for_display(profile):
    """Format profile information for Streamlit display."""
    overview = {
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Features (excl. target)",
            "Target Column",
            "Number of Classes",
            "Numerical Features",
            "Categorical Features",
            "Total Missing Values",
            "Missing Value Ratio",
            "Duplicate Rows",
            "Class Imbalance Ratio",
            "Memory Usage"
        ],
        "Value": [
            f"{profile['n_rows']:,}",
            str(profile['n_cols']),
            str(profile['n_features']),
            profile['target_col'],
            str(profile['n_classes']),
            str(profile['numerical_features']),
            str(profile['categorical_features']),
            f"{profile['total_missing']:,}",
            f"{profile['missing_ratio']}%",
            str(profile['duplicate_rows']),
            str(profile['class_imbalance_ratio']),
            f"{profile['memory_mb']} MB"
        ]
    }
    return pd.DataFrame(overview)
