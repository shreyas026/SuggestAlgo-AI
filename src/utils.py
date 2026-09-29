"""
SuggestAlgo AI - Utility Functions

Helper functions used across the application.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
import os


def get_sample_datasets():
    """Get available sample datasets with detailed metadata dictionary."""
    return {
        'Iris Dataset': {
            'id': 'iris',
            'description': 'Classification - 3 classes, 150 samples',
            'target': 'species',
            'task': 'classification'
        },
        'Breast Cancer Dataset': {
            'id': 'breast_cancer',
            'description': 'Binary Classification - 2 classes, 569 samples',
            'target': 'diagnosis',
            'task': 'classification'
        },
        'Wine Dataset': {
            'id': 'wine',
            'description': 'Classification - 3 classes, 178 samples',
            'target': 'wine_class',
            'task': 'classification'
        },
        'Diabetes Dataset': {
            'id': 'diabetes',
            'description': 'Regression - 442 samples',
            'target': 'target',
            'task': 'regression'
        }
    }


def load_sample_dataset(dataset_name):
    """Load a sample dataset and return (df, target_column)."""
    name_clean = str(dataset_name).lower()
    
    if 'iris' in name_clean:
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['species'] = pd.Categorical.from_codes(data.target, data.target_names)
        return df, 'species'
    
    elif 'breast_cancer' in name_clean or 'cancer' in name_clean:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = pd.Categorical.from_codes(data.target, data.target_names)
        return df, 'diagnosis'
    
    elif 'wine' in name_clean:
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['wine_class'] = data.target
        return df, 'wine_class'
    
    elif 'diabetes' in name_clean:
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target'
    
    return None, None


def save_sample_datasets(output_dir):
    """Save sample datasets as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        'iris': load_iris,
        'breast_cancer': load_breast_cancer,
        'wine': load_wine,
    }
    
    for name, loader in datasets.items():
        data = loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        filepath = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(filepath, index=False)
    
    return True


def format_metric(value, decimals=4):
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def get_model_description(model_name):
    """Get a brief description for each candidate algorithm."""
    descriptions = {
        'RandomForest': (
            "An ensemble of decision trees trained on random subsets of data and features. "
            "Robust against overfitting and handles high-dimensional data well."
        ),
        'DecisionTree': (
            "A tree-structured model that splits data based on feature thresholds. "
            "Highly interpretable but prone to overfitting without pruning."
        ),
        'LogisticRegression': (
            "A linear model for classification using logistic function. "
            "Fast, interpretable, and works well when features are linearly separable."
        ),
        'SVC': (
            "Finds the optimal hyperplane to separate classes in feature space. "
            "Effective in high-dimensional spaces, especially with kernel trick."
        ),
        'ExtraTrees': (
            "Similar to Random Forest but uses random thresholds for splitting. "
            "Often faster and can reduce variance further."
        ),
        'GradientBoosting': (
            "Builds trees sequentially, each correcting errors of the previous. "
            "Often achieves high accuracy but can be slower to train."
        ),
        'AdaBoost': (
            "An adaptive boosting method that focuses on misclassified samples. "
            "Simple to implement and less prone to overfitting than other boosters."
        ),
        'XGBoost': (
            "Optimized gradient boosting with regularization. "
            "State-of-the-art performance on many structured/tabular datasets."
        ),
        'SGDClassifier': (
            "Linear classifier optimized with stochastic gradient descent. "
            "Very fast and memory-efficient, suitable for large-scale learning."
        ),
    }
    return descriptions.get(model_name, "No description available.")
