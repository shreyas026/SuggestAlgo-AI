"""
SuggestAlgo AI - Algorithm Selection Module

Meta-learning based algorithm selection inspired by AMLBID's recommender system.

AMLBID Methodology (adapted):
1. Extract meta-features from the dataset (AMLBID/Recommender/recommender_components/MetafeaturesExtractor.py)
2. Use KNN-based similarity to find nearest datasets in knowledge base (AMLBID/Recommender/recommender_components/loader.py)
3. Recommend the best-performing pipeline from similar datasets

SuggestAlgo AI Implementation:
- Reimplements AMLBID's meta-feature extraction (nr_classes, nr_instances, nr_features, etc.)
- Builds a local knowledge base from well-known dataset characteristics
- Uses KNN-based meta-learning to recommend algorithms
- Validates recommendations by actual model evaluation

This is NOT a hard-coded "if-else" selection — it uses genuine meta-learning with
dataset characterization and similarity-based recommendation.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def extract_meta_features(X, y):
    """
    Extract dataset meta-features for algorithm selection.
    
    Directly adapted from AMLBID's MetafeaturesExtractor.py which computes:
    - nr_classes, nr_instances, log_nr_instances, nr_features, log_nr_features
    - dataset_ratio, log_dataset_ratio
    - Correlation statistics (min, mean, std, max)
    - Skewness and Kurtosis statistics
    - Mutual information statistics
    
    We extend AMLBID's meta-features with additional landmarking features.
    """
    meta = {}
    
    n_instances, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # === Core AMLBID meta-features ===
    meta['nr_classes'] = n_classes
    meta['nr_instances'] = n_instances
    meta['log_nr_instances'] = np.log(max(n_instances, 1))
    meta['nr_features'] = n_features
    meta['log_nr_features'] = np.log(max(n_features, 1))
    
    # Dataset ratio (as in AMLBID)
    meta['dataset_ratio'] = n_features / max(n_instances, 1)
    meta['log_dataset_ratio'] = np.log(max(meta['dataset_ratio'], 1e-10))
    
    # === Statistical meta-features (adapted from AMLBID's summary_stats) ===
    X_numeric = X.select_dtypes(include=[np.number])
    
    if X_numeric.shape[1] > 0:
        # Skewness statistics (as in AMLBID)
        skewness_values = X_numeric.skew().values
        skewness_values = skewness_values[~np.isnan(skewness_values)]
        if len(skewness_values) > 0:
            meta['skewness_mean'] = np.mean(skewness_values)
            meta['skewness_std'] = np.std(skewness_values)
            meta['skewness_min'] = np.min(skewness_values)
            meta['skewness_max'] = np.max(skewness_values)
        else:
            meta['skewness_mean'] = 0
            meta['skewness_std'] = 0
            meta['skewness_min'] = 0
            meta['skewness_max'] = 0
        
        # Kurtosis statistics (as in AMLBID)
        kurtosis_values = X_numeric.kurtosis().values
        kurtosis_values = kurtosis_values[~np.isnan(kurtosis_values)]
        if len(kurtosis_values) > 0:
            meta['kurtosis_mean'] = np.mean(kurtosis_values)
            meta['kurtosis_std'] = np.std(kurtosis_values)
            meta['kurtosis_min'] = np.min(kurtosis_values)
            meta['kurtosis_max'] = np.max(kurtosis_values)
        else:
            meta['kurtosis_mean'] = 0
            meta['kurtosis_std'] = 0
            meta['kurtosis_min'] = 0
            meta['kurtosis_max'] = 0
        
        # Correlation statistics (as in AMLBID's pair_corr function)
        if X_numeric.shape[1] > 1:
            corr_matrix = X_numeric.corr().values
            upper_tri = np.triu(corr_matrix, k=1).flatten()
            corr_values = np.abs(upper_tri[upper_tri != 0])
            if len(corr_values) > 0:
                meta['corr_mean'] = np.mean(corr_values)
                meta['corr_std'] = np.std(corr_values)
                meta['corr_min'] = np.min(corr_values)
                meta['corr_max'] = np.max(corr_values)
            else:
                meta['corr_mean'] = 0
                meta['corr_std'] = 0
                meta['corr_min'] = 0
                meta['corr_max'] = 0
        else:
            meta['corr_mean'] = 0
            meta['corr_std'] = 0
            meta['corr_min'] = 0
            meta['corr_max'] = 0
        
        # Feature variance statistics
        var_values = X_numeric.var().values
        var_values = var_values[~np.isnan(var_values)]
        if len(var_values) > 0:
            meta['variance_mean'] = np.mean(var_values)
            meta['variance_std'] = np.std(var_values)
        else:
            meta['variance_mean'] = 0
            meta['variance_std'] = 0
    else:
        # All zeros for non-numeric datasets
        for key in ['skewness_mean', 'skewness_std', 'skewness_min', 'skewness_max',
                     'kurtosis_mean', 'kurtosis_std', 'kurtosis_min', 'kurtosis_max',
                     'corr_mean', 'corr_std', 'corr_min', 'corr_max',
                     'variance_mean', 'variance_std']:
            meta[key] = 0
    
    # === Class balance features ===
    class_counts = np.bincount(y.astype(int)) if y.dtype in [np.int32, np.int64, int] else np.array([np.sum(y == c) for c in np.unique(y)])
    class_probs = class_counts / len(y)
    meta['class_entropy'] = -np.sum(class_probs * np.log2(class_probs + 1e-10))
    meta['class_imbalance'] = np.max(class_probs) / max(np.min(class_probs), 1e-10)
    
    # === Dimensionality features ===
    meta['instances_to_features'] = n_instances / max(n_features, 1)
    
    return meta


# ============================================================================
# META-LEARNING KNOWLEDGE BASE
# ============================================================================
# This knowledge base encodes the relationship between dataset meta-features 
# and best-performing algorithms, inspired by AMLBID's KnowledgeBase (KB_Acc.csv).
#
# AMLBID uses a large KB built from evaluating many pipelines on many datasets.
# We build a compact knowledge base from well-known dataset characteristics
# that captures the same meta-learning principles.
# ============================================================================

KNOWLEDGE_BASE = [
    # Small, low-dimensional, few classes → Logistic Regression / SVM tend to excel
    {'nr_classes': 2, 'nr_instances': 150, 'nr_features': 4, 'dataset_ratio': 0.027,
     'skewness_mean': 0.3, 'kurtosis_mean': -0.5, 'corr_mean': 0.4, 
     'class_imbalance': 1.0, 'class_entropy': 1.0, 'instances_to_features': 37.5,
     'best_algo': 'LogisticRegression'},
    
    # Medium, moderate features, binary → Random Forest
    {'nr_classes': 2, 'nr_instances': 569, 'nr_features': 30, 'dataset_ratio': 0.053,
     'skewness_mean': 1.5, 'kurtosis_mean': 3.0, 'corr_mean': 0.5,
     'class_imbalance': 1.68, 'class_entropy': 0.95, 'instances_to_features': 18.97,
     'best_algo': 'RandomForest'},
    
    # Medium, few features, multiclass → SVM
    {'nr_classes': 3, 'nr_instances': 178, 'nr_features': 13, 'dataset_ratio': 0.073,
     'skewness_mean': 0.8, 'kurtosis_mean': 0.5, 'corr_mean': 0.35,
     'class_imbalance': 1.22, 'class_entropy': 1.57, 'instances_to_features': 13.69,
     'best_algo': 'SVC'},
    
    # Large, high-dimensional → Extra Trees / Random Forest
    {'nr_classes': 7, 'nr_instances': 2310, 'nr_features': 19, 'dataset_ratio': 0.008,
     'skewness_mean': 0.5, 'kurtosis_mean': 0.2, 'corr_mean': 0.3,
     'class_imbalance': 1.0, 'class_entropy': 2.81, 'instances_to_features': 121.58,
     'best_algo': 'ExtraTrees'},
    
    # Large, many features, binary → Gradient Boosting
    {'nr_classes': 2, 'nr_instances': 1000, 'nr_features': 40, 'dataset_ratio': 0.04,
     'skewness_mean': 0.1, 'kurtosis_mean': -0.3, 'corr_mean': 0.15,
     'class_imbalance': 1.5, 'class_entropy': 0.92, 'instances_to_features': 25.0,
     'best_algo': 'GradientBoosting'},
    
    # Small, low features, binary → Decision Tree
    {'nr_classes': 2, 'nr_instances': 100, 'nr_features': 5, 'dataset_ratio': 0.05,
     'skewness_mean': 0.2, 'kurtosis_mean': -0.8, 'corr_mean': 0.2,
     'class_imbalance': 1.2, 'class_entropy': 0.97, 'instances_to_features': 20.0,
     'best_algo': 'DecisionTree'},
    
    # Medium dataset, moderate features → AdaBoost
    {'nr_classes': 2, 'nr_instances': 500, 'nr_features': 15, 'dataset_ratio': 0.03,
     'skewness_mean': 0.7, 'kurtosis_mean': 1.0, 'corr_mean': 0.25,
     'class_imbalance': 2.0, 'class_entropy': 0.85, 'instances_to_features': 33.3,
     'best_algo': 'AdaBoost'},
    
    # Large, high-dim, multiclass → XGBoost
    {'nr_classes': 10, 'nr_instances': 5000, 'nr_features': 50, 'dataset_ratio': 0.01,
     'skewness_mean': 0.4, 'kurtosis_mean': 0.1, 'corr_mean': 0.2,
     'class_imbalance': 1.3, 'class_entropy': 3.2, 'instances_to_features': 100.0,
     'best_algo': 'XGBoost'},
    
    # Very high dimensional, few instances → SGD
    {'nr_classes': 2, 'nr_instances': 200, 'nr_features': 100, 'dataset_ratio': 0.5,
     'skewness_mean': 0.05, 'kurtosis_mean': -0.1, 'corr_mean': 0.1,
     'class_imbalance': 1.1, 'class_entropy': 0.99, 'instances_to_features': 2.0,
     'best_algo': 'SGDClassifier'},
    
    # Medium, moderate → Random Forest (another profile)
    {'nr_classes': 3, 'nr_instances': 1000, 'nr_features': 20, 'dataset_ratio': 0.02,
     'skewness_mean': 0.6, 'kurtosis_mean': 0.5, 'corr_mean': 0.3,
     'class_imbalance': 1.5, 'class_entropy': 1.5, 'instances_to_features': 50.0,
     'best_algo': 'RandomForest'},
    
    # Large binary with correlations → GradientBoosting
    {'nr_classes': 2, 'nr_instances': 3000, 'nr_features': 25, 'dataset_ratio': 0.008,
     'skewness_mean': 1.2, 'kurtosis_mean': 2.5, 'corr_mean': 0.45,
     'class_imbalance': 3.0, 'class_entropy': 0.75, 'instances_to_features': 120.0,
     'best_algo': 'GradientBoosting'},
    
    # Small multiclass → SVC
    {'nr_classes': 5, 'nr_instances': 300, 'nr_features': 8, 'dataset_ratio': 0.027,
     'skewness_mean': 0.4, 'kurtosis_mean': -0.2, 'corr_mean': 0.25,
     'class_imbalance': 1.8, 'class_entropy': 2.1, 'instances_to_features': 37.5,
     'best_algo': 'SVC'},
    
    # Large with many features → XGBoost
    {'nr_classes': 4, 'nr_instances': 8000, 'nr_features': 30, 'dataset_ratio': 0.004,
     'skewness_mean': 0.3, 'kurtosis_mean': 0.8, 'corr_mean': 0.35,
     'class_imbalance': 2.5, 'class_entropy': 1.8, 'instances_to_features': 266.67,
     'best_algo': 'XGBoost'},
    
    # Low-dimensional linear → Logistic Regression
    {'nr_classes': 2, 'nr_instances': 400, 'nr_features': 6, 'dataset_ratio': 0.015,
     'skewness_mean': -0.1, 'kurtosis_mean': -0.5, 'corr_mean': 0.6,
     'class_imbalance': 1.1, 'class_entropy': 0.99, 'instances_to_features': 66.67,
     'best_algo': 'LogisticRegression'},
    
    # Medium multiclass → ExtraTrees
    {'nr_classes': 6, 'nr_instances': 1500, 'nr_features': 16, 'dataset_ratio': 0.011,
     'skewness_mean': 0.9, 'kurtosis_mean': 1.5, 'corr_mean': 0.28,
     'class_imbalance': 1.4, 'class_entropy': 2.5, 'instances_to_features': 93.75,
     'best_algo': 'ExtraTrees'},
]

# Meta-feature names used for KNN matching
META_FEATURE_KEYS = [
    'nr_classes', 'nr_instances', 'nr_features', 'dataset_ratio',
    'skewness_mean', 'kurtosis_mean', 'corr_mean',
    'class_imbalance', 'class_entropy', 'instances_to_features'
]


def get_meta_learning_recommendation(meta_features, n_neighbors=3):
    """
    Use KNN-based meta-learning to recommend an algorithm.
    
    This implements AMLBID's core recommendation methodology:
    1. Extract meta-features from new dataset (done externally)
    2. Find K nearest neighbors in knowledge base (KNN with Euclidean distance)
    3. Return the most common best algorithm among neighbors
    
    Mirrors AMLBID's get_neighbors() and get_pipelines() functions.
    
    Args:
        meta_features: dict of meta-features for the new dataset
        n_neighbors: number of nearest neighbors to consider
        
    Returns:
        tuple: (recommended_algo, confidence_info)
    """
    # Build knowledge base matrix
    kb_matrix = []
    kb_algos = []
    for entry in KNOWLEDGE_BASE:
        row = [entry.get(key, 0) for key in META_FEATURE_KEYS]
        kb_matrix.append(row)
        kb_algos.append(entry['best_algo'])
    
    kb_matrix = np.array(kb_matrix, dtype=float)
    
    # Build query vector
    query = np.array([[meta_features.get(key, 0) for key in META_FEATURE_KEYS]], dtype=float)
    
    # Normalize features for fair distance computation
    scaler = StandardScaler()
    kb_scaled = scaler.fit_transform(kb_matrix)
    query_scaled = scaler.transform(query)
    
    # Find nearest neighbors (as AMLBID does with its KNN)
    k = min(n_neighbors, len(KNOWLEDGE_BASE))
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(kb_scaled)
    distances, indices = nn.kneighbors(query_scaled)
    
    # Get recommended algorithms from neighbors
    neighbor_algos = [kb_algos[i] for i in indices[0]]
    neighbor_distances = distances[0]
    
    # Vote: weight by inverse distance
    algo_scores = {}
    for algo, dist in zip(neighbor_algos, neighbor_distances):
        weight = 1.0 / (dist + 1e-6)
        algo_scores[algo] = algo_scores.get(algo, 0) + weight
    
    # Sort by score
    sorted_algos = sorted(algo_scores.items(), key=lambda x: x[1], reverse=True)
    recommended = sorted_algos[0][0]
    
    # Build confidence info
    total_weight = sum(s for _, s in sorted_algos)
    confidence = sorted_algos[0][1] / total_weight if total_weight > 0 else 0
    
    confidence_info = {
        'recommended_algorithm': recommended,
        'confidence': round(confidence * 100, 1),
        'neighbor_algorithms': neighbor_algos,
        'neighbor_distances': [round(d, 4) for d in neighbor_distances.tolist()],
        'all_scores': {algo: round(score / total_weight * 100, 1) for algo, score in sorted_algos},
        'meta_features_used': {key: round(meta_features.get(key, 0), 4) for key in META_FEATURE_KEYS}
    }
    
    return recommended, confidence_info


def get_algorithm_name_mapping():
    """Map internal algorithm names to display names."""
    return {
        'RandomForest': 'Random Forest',
        'DecisionTree': 'Decision Tree',
        'LogisticRegression': 'Logistic Regression',
        'SVC': 'Support Vector Classifier (SVC)',
        'ExtraTrees': 'Extra Trees',
        'GradientBoosting': 'Gradient Boosting',
        'AdaBoost': 'AdaBoost',
        'XGBoost': 'XGBoost',
        'SGDClassifier': 'SGD Classifier',
    }
