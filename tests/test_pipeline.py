"""
SuggestAlgo AI - Basic Tests

Tests core pipeline components: data loading, preprocessing, 
meta-feature extraction, algorithm selection, and model evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from src.data_loader import detect_column_types, get_target_candidates, validate_target
from src.preprocessing import preprocess_dataset
from src.profiling import profile_dataset, format_profile_for_display
from src.algorithm_selection import extract_meta_features, get_meta_learning_recommendation
from src.model_evaluation import evaluate_all_models, get_candidate_models
from src.utils import load_sample_dataset


def test_data_loading():
    """Test dataset loading and validation."""
    print("Testing data loading...", end=" ")
    
    df, target = load_sample_dataset('iris')
    assert df is not None, "Failed to load iris dataset"
    assert df.shape[0] == 150, f"Expected 150 rows, got {df.shape[0]}"
    assert target == 'species', f"Expected 'species' target, got {target}"
    
    num_cols, cat_cols = detect_column_types(df)
    assert len(num_cols) == 4, f"Expected 4 numerical cols, got {len(num_cols)}"
    assert len(cat_cols) == 1, f"Expected 1 categorical col, got {len(cat_cols)}"
    
    candidates = get_target_candidates(df)
    assert 'species' in candidates, "species should be a target candidate"
    
    errors = validate_target(df, 'species')
    assert len(errors) == 0, f"Target validation should pass, got errors: {errors}"
    
    print("PASSED ✅")


def test_profiling():
    """Test dataset profiling."""
    print("Testing profiling...", end=" ")
    
    df, target = load_sample_dataset('iris')
    profile = profile_dataset(df, target)
    
    assert profile['n_rows'] == 150
    assert profile['n_classes'] == 3
    assert profile['target_col'] == 'species'
    assert profile['total_missing'] == 0
    assert profile['duplicate_rows'] >= 0
    
    display_df = format_profile_for_display(profile)
    assert len(display_df) > 0, "Display DataFrame should not be empty"
    
    print("PASSED ✅")


def test_preprocessing():
    """Test data preprocessing."""
    print("Testing preprocessing...", end=" ")
    
    df, target = load_sample_dataset('iris')
    X, y, feature_names, target_le, prep_info = preprocess_dataset(df, target)
    
    assert X.shape[0] == 150, f"Expected 150 samples, got {X.shape[0]}"
    assert X.shape[1] == 4, f"Expected 4 features, got {X.shape[1]}"
    assert len(y) == 150
    assert len(np.unique(y)) == 3
    assert prep_info['n_classes'] == 3
    
    print("PASSED ✅")


def test_meta_features():
    """Test meta-feature extraction."""
    print("Testing meta-feature extraction...", end=" ")
    
    df, target = load_sample_dataset('iris')
    X, y, feature_names, target_le, prep_info = preprocess_dataset(df, target)
    
    meta = extract_meta_features(X, y)
    
    assert meta['nr_instances'] == 150
    assert meta['nr_features'] == 4
    assert meta['nr_classes'] == 3
    assert 'skewness_mean' in meta
    assert 'kurtosis_mean' in meta
    assert 'corr_mean' in meta
    assert 'class_entropy' in meta
    assert 'dataset_ratio' in meta
    
    print("PASSED ✅")


def test_algorithm_selection():
    """Test meta-learning algorithm selection."""
    print("Testing algorithm selection...", end=" ")
    
    df, target = load_sample_dataset('iris')
    X, y, feature_names, target_le, prep_info = preprocess_dataset(df, target)
    meta = extract_meta_features(X, y)
    
    recommended, confidence_info = get_meta_learning_recommendation(meta)
    
    assert recommended is not None, "Should recommend an algorithm"
    assert isinstance(recommended, str)
    assert confidence_info['confidence'] > 0, "Confidence should be positive"
    assert len(confidence_info['neighbor_algorithms']) > 0
    
    print(f"PASSED ✅ (Recommended: {recommended}, Confidence: {confidence_info['confidence']:.1f}%)")


def test_model_evaluation():
    """Test model training and evaluation."""
    print("Testing model evaluation...", end=" ")
    
    df, target = load_sample_dataset('iris')
    X, y, feature_names, target_le, prep_info = preprocess_dataset(df, target)
    
    results, comparison_df, best_model, split_data = evaluate_all_models(X, y)
    
    # At least some models should succeed
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    assert successful >= 5, f"At least 5 models should succeed, got {successful}"
    
    # Best model should be identified
    assert best_model is not None, "Best model should be identified"
    
    # Check metrics are real (not fake/hardcoded)
    best_result = results[best_model]
    assert 0 < best_result['accuracy'] <= 1.0, "Accuracy should be between 0 and 1"
    assert 0 < best_result['f1_score'] <= 1.0, "F1 should be between 0 and 1"
    assert best_result['training_time'] > 0, "Training time should be positive"
    
    # Comparison DataFrame should have data
    assert len(comparison_df) > 0
    
    print(f"PASSED ✅ (Best: {best_model}, F1: {best_result['f1_score']:.4f})")


def test_breast_cancer():
    """Test with breast cancer dataset (binary classification)."""
    print("Testing with breast cancer dataset...", end=" ")
    
    df, target = load_sample_dataset('breast_cancer')
    X, y, feature_names, target_le, prep_info = preprocess_dataset(df, target)
    meta = extract_meta_features(X, y)
    recommended, confidence_info = get_meta_learning_recommendation(meta)
    results, comparison_df, best_model, split_data = evaluate_all_models(X, y)
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    assert successful >= 5
    
    print(f"PASSED ✅ (Recommended: {recommended}, Best: {best_model})")


def test_wine():
    """Test with wine dataset (multiclass)."""
    print("Testing with wine dataset...", end=" ")
    
    df, target = load_sample_dataset('wine')
    X, y, feature_names, target_le, prep_info = preprocess_dataset(df, target)
    meta = extract_meta_features(X, y)
    recommended, confidence_info = get_meta_learning_recommendation(meta)
    results, comparison_df, best_model, split_data = evaluate_all_models(X, y)
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    assert successful >= 5
    
    print(f"PASSED ✅ (Recommended: {recommended}, Best: {best_model})")


def test_error_handling():
    """Test error handling with edge cases."""
    print("Testing error handling...", end=" ")
    
    # Single-class target
    errors = validate_target(pd.DataFrame({'a': [1,2,3], 'b': [1,1,1]}), 'b')
    assert len(errors) > 0, "Should error on single-class target"
    
    # Too many unique values
    errors = validate_target(pd.DataFrame({'a': range(200), 'b': range(200)}), 'b')
    assert len(errors) > 0, "Should error on too many classes"
    
    print("PASSED ✅")


def test_nlp_mode2():
    """Test Mode 2 natural-language problem & project idea algorithm recommendations."""
    print("Testing Mode 2 NLP algorithm recommendation...", end=" ")
    from src.nlp_recommendation import recommend_algorithm_from_text
    
    # 1. Coding problem search test
    res_search = recommend_algorithm_from_text("I have a sorted array with 1M elements and need to find if a target exists")
    assert res_search['top_recommendation'] is not None
    assert res_search['top_score'] > 60.0
    
    # 2. Graph shortest path test
    res_graph = recommend_algorithm_from_text("Find the shortest path between two cities in a weighted graph network")
    assert res_graph['top_recommendation'] is not None
    assert "Dijkstra" in res_graph['top_recommendation']['name'] or "Graph" in res_graph['top_recommendation']['category']
    
    # 3. Machine Learning project idea test
    res_churn = recommend_algorithm_from_text("Predict customer churn for an e-commerce subscription dataset")
    assert res_churn['analysis']['is_ml_project'] == True
    assert res_churn['top_recommendation'] is not None
    
    # 4. Vague input follow-up questions test
    res_vague = recommend_algorithm_from_text("I want to build a project")
    assert res_vague['analysis']['is_vague'] == True
    assert len(res_vague['analysis']['follow_up_questions']) > 0
    
    print("PASSED ✅")


if __name__ == "__main__":
    print("=" * 60)
    print("SuggestAlgo AI — Running Tests")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_profiling,
        test_preprocessing,
        test_meta_features,
        test_algorithm_selection,
        test_model_evaluation,
        test_breast_cancer,
        test_wine,
        test_error_handling,
        test_nlp_mode2,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED ❌ ({str(e)})")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
