"""
SuggestAlgo AI - Data Loader Module

Handles CSV file loading, validation, and basic data quality checks.
Inspired by AMLBID's loader.py preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import io


def validate_file(uploaded_file):
    """Validate the uploaded file before processing."""
    errors = []
    
    if uploaded_file is None:
        errors.append("No file uploaded.")
        return None, errors
    
    # Check file size (max 200MB)
    file_size = uploaded_file.size
    if file_size > 200 * 1024 * 1024:
        errors.append("File size exceeds 200MB limit.")
        return None, errors
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.csv'):
        errors.append("Only CSV files are supported. Please upload a .csv file.")
        return None, errors
    
    return uploaded_file, errors


def load_dataset(uploaded_file):
    """
    Load a CSV dataset with robust error handling.
    
    Adapted from AMLBID's loader.preprocess() function which handles
    CSV reading with flexible separators and basic preprocessing.
    
    Returns:
        tuple: (DataFrame, list of errors)
    """
    errors = []
    
    try:
        # Try common separators - AMLBID uses sep='[;,]' with python engine
        try:
            df = pd.read_csv(uploaded_file, sep=',', engine='python')
        except Exception:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=';', engine='python')
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep='\t', engine='python')
        
        if df.empty:
            errors.append("The uploaded CSV file is empty.")
            return None, errors
        
        if df.shape[0] < 5:
            errors.append("Dataset must contain at least 5 rows for meaningful analysis.")
            return None, errors
        
        if df.shape[1] < 2:
            errors.append("Dataset must contain at least 2 columns (features + target).")
            return None, errors
        
        return df, errors
        
    except pd.errors.EmptyDataError:
        errors.append("The uploaded file is empty or contains no parseable data.")
        return None, errors
    except pd.errors.ParserError as e:
        errors.append(f"Could not parse the CSV file: {str(e)}")
        return None, errors
    except Exception as e:
        errors.append(f"Error loading dataset: {str(e)}")
        return None, errors


def detect_column_types(df):
    """Detect numerical and categorical columns in the dataset."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerical_cols, categorical_cols


def get_target_candidates(df):
    """
    Identify columns suitable as classification targets.
    A good target column typically has a limited number of unique values.
    """
    candidates = []
    for col in df.columns:
        n_unique = df[col].nunique()
        n_rows = len(df)
        # Target should have at least 2 classes but not too many relative to dataset size
        if 2 <= n_unique <= min(50, n_rows // 5):
            candidates.append(col)
    
    # If no good candidates found, return all columns
    if not candidates:
        candidates = df.columns.tolist()
    
    return candidates


def validate_target(df, target_col):
    """Validate the selected target column for classification."""
    errors = []
    
    if target_col not in df.columns:
        errors.append(f"Column '{target_col}' not found in dataset.")
        return errors
    
    n_unique = df[target_col].nunique()
    
    if n_unique < 2:
        errors.append(f"Target column '{target_col}' has only {n_unique} unique value(s). "
                      "Classification requires at least 2 classes.")
    
    if n_unique > 100:
        errors.append(f"Target column '{target_col}' has {n_unique} unique values. "
                      "This appears to be a continuous variable, not suitable for classification. "
                      "Please select a categorical target column.")
    
    # Check for excessive missing values in target
    missing_pct = df[target_col].isnull().mean() * 100
    if missing_pct > 50:
        errors.append(f"Target column '{target_col}' has {missing_pct:.1f}% missing values. "
                      "This is too many for reliable classification.")
    
    return errors
