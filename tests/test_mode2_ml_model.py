"""
SuggestAlgo AI - Test Suite for Supervised Mode 2 Algorithm Classifier

Verifies:
1. Dataset creation & schema
2. Train/Test split & zero leakage
3. TF-IDF + Logistic Regression pipeline
4. Model save / load
5. Prediction & Top-K category filtering
6. Evaluation metrics & output files
"""

import sys
import os
import json
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.mode2_ml_model import (
    MODEL_PATH, DATASET_PATH, TRAIN_PATH, TEST_PATH,
    RESULTS_JSON_PATH, CONFUSION_MATRIX_PATH,
    load_mode2_model, predict_algorithm, predict_top_k_algorithms,
    train_mode2_model
)


def test_1_dataset_creation_and_schema():
    """Verify curated dataset file exists and has correct columns and row count."""
    assert os.path.exists(DATASET_PATH), f"Dataset file {DATASET_PATH} not found"
    df = pd.read_csv(DATASET_PATH)
    assert len(df) == 5600, f"Expected 5600 rows, got {len(df)}"
    
    required_cols = [
        "problem_id", "problem_description", "category", "algorithm",
        "data_structures", "input_characteristics", "constraints", "optimization_goal"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}'"
        
    assert df['algorithm'].nunique() == 112, f"Expected 112 unique algorithms, got {df['algorithm'].nunique()}"


def test_2_train_val_test_split_and_no_leakage():
    """Verify 3-way dataset split exists with zero text overlap across train, val, and test sets."""
    data_dir = os.path.dirname(DATASET_PATH)
    train_p = os.path.join(data_dir, 'train_algorithm_dataset.csv')
    val_p = os.path.join(data_dir, 'validation_algorithm_dataset.csv')
    test_p = os.path.join(data_dir, 'test_algorithm_dataset.csv')
    
    assert os.path.exists(train_p), f"Train dataset {train_p} not found"
    assert os.path.exists(val_p), f"Val dataset {val_p} not found"
    assert os.path.exists(test_p), f"Test dataset {test_p} not found"
    
    train_df = pd.read_csv(train_p)
    val_df = pd.read_csv(val_p)
    test_df = pd.read_csv(test_p)
    
    assert len(train_df) == 3920, f"Expected 3920 train rows, got {len(train_df)}"
    assert len(val_df) == 840, f"Expected 840 val rows, got {len(val_df)}"
    assert len(test_df) == 840, f"Expected 840 test rows, got {len(test_df)}"
    
    train_texts = set(train_df['problem_description'])
    val_texts = set(val_df['problem_description'])
    test_texts = set(test_df['problem_description'])
    
    assert len(train_texts.intersection(val_texts)) == 0, "Leakage between train and val"
    assert len(train_texts.intersection(test_texts)) == 0, "Leakage between train and test"
    assert len(val_texts.intersection(test_texts)) == 0, "Leakage between val and test"


def test_3_model_save_and_load():
    """Verify saved joblib pipeline exists and loads cleanly."""
    assert os.path.exists(MODEL_PATH), f"Model file {MODEL_PATH} not found"
    model = load_mode2_model()
    assert model is not None
    assert hasattr(model, 'predict')


def test_4_single_prediction():
    """Verify single algorithm prediction for sample problem statement."""
    prompt = "I have a sorted list of numbers and want to find an element in logarithmic time."
    algo = predict_algorithm(prompt)
    assert isinstance(algo, str)
    assert len(algo) > 0


def test_5_top_k_prediction_and_category_filtering():
    """Verify predict_top_k_algorithms with category filtering."""
    prompt = "Find shortest path between cities in a weighted graph network"
    preds = predict_top_k_algorithms(prompt, selected_categories=["Graph Algorithms"], k=3)
    
    assert len(preds) <= 3
    assert len(preds) > 0
    for p in preds:
        assert p['category'] == "Graph Algorithms"
        assert 'algorithm' in p
        assert 'ml_probability' in p
        assert 0.0 <= p['ml_probability'] <= 100.0


def test_6_evaluation_metrics_and_artifacts():
    """Verify evaluation results JSON and confusion matrix plot exist."""
    assert os.path.exists(RESULTS_JSON_PATH), f"Results JSON {RESULTS_JSON_PATH} not found"
    assert os.path.exists(CONFUSION_MATRIX_PATH), f"Confusion matrix plot {CONFUSION_MATRIX_PATH} not found"
    
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)
        
    assert res['accuracy'] >= 0.70, f"Accuracy {res['accuracy']} lower than expected threshold 0.70"
    assert res['num_algorithms'] == 112
    assert res['test_samples'] == 840


if __name__ == "__main__":
    pytest.main([__file__])
