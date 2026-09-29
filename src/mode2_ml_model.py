"""
SuggestAlgo AI - Supervised Mode 2 Algorithm Recommendation ML Model

Pipeline:
Problem Description -> TF-IDF Vectorizer (1-2 N-Grams) -> Logistic Regression -> Category-Filtered Algorithm Prediction

Trained on curated problem-algorithm benchmark dataset derived from the algorithm knowledge base.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from src.algorithm_knowledge_base import ALGORITHM_KNOWLEDGE_BASE

MODEL_PATH = "models/mode2_algorithm_classifier.joblib"
DATASET_PATH = "data/problem_algorithm_dataset.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
RESULTS_JSON_PATH = "results/mode2_ml_results.json"
RESULTS_CSV_PATH = "results/mode2_metrics.csv"
CONFUSION_MATRIX_PATH = "results/mode2_confusion_matrix.png"

# Mapping algorithm name to category
ALGO_TO_CATEGORY = {algo["name"]: algo["category"] for algo in ALGORITHM_KNOWLEDGE_BASE}
_CACHED_MODEL = None


def create_train_test_split(dataset_path=DATASET_PATH, random_state=42):
    """
    Split problem-algorithm dataset into 80% train and 20% test stratified by algorithm.
    Saves data/train.csv and data/test.csv.
    """
    df = pd.read_csv(dataset_path)
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        stratify=df['algorithm']
    )
    
    os.makedirs("data", exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    
    return train_df, test_df


def train_mode2_model(random_state=42):
    """
    Train TF-IDF + LogisticRegression Pipeline on train.csv and evaluate on unseen test.csv.
    Saves trained pipeline to models/mode2_algorithm_classifier.joblib and outputs metrics.
    """
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        train_df, test_df = create_train_test_split(random_state=random_state)
    else:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
    X_train = train_df['problem_description']
    y_train = train_df['algorithm']
    
    X_test = test_df['problem_description']
    y_test = test_df['algorithm']
    
    # Define Pipeline - TF-IDF fit ONLY on training data
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=3000,
            random_state=random_state
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    
    # Evaluate on unseen test set
    metrics, cm = evaluate_mode2_model(pipeline, X_test, y_test, train_df, test_df)
    
    global _CACHED_MODEL
    _CACHED_MODEL = pipeline
    
    return pipeline, metrics


def evaluate_mode2_model(model, X_test, y_test, train_df, test_df):
    """
    Calculate classification metrics on unseen test set and save to results/mode2_ml_results.json.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "dataset_type": "Curated/Synthetic Algorithm Problem Benchmark",
        "dataset_source": "SuggestAlgo AI Algorithm Knowledge Base",
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "num_algorithms": int(len(model.classes_)),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "random_state": 42
    }
    
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(RESULTS_CSV_PATH, index=False)
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Mode 2 Algorithm Classifier — Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Algorithm')
    plt.ylabel('True Algorithm')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    
    return metrics, cm


def load_mode2_model():
    """Load or train and return the Mode 2 ML pipeline model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
        
    if os.path.exists(MODEL_PATH):
        try:
            _CACHED_MODEL = joblib.load(MODEL_PATH)
            return _CACHED_MODEL
        except Exception:
            pass
            
    # Train if model doesn't exist
    model, _ = train_mode2_model()
    return model


def predict_algorithm(problem_text):
    """Predict top single algorithm name for a problem text."""
    model = load_mode2_model()
    pred = model.predict([problem_text])[0]
    return pred


def predict_top_k_algorithms(problem_text, selected_categories=None, k=3):
    """
    Predict top-K algorithm candidates for problem_text with category filtering.
    
    Returns list of dicts:
    [
      {
        "algorithm": algo_name,
        "category": cat_name,
        "ml_probability": float_percentage,
        "rank": int
      }, ...
    ]
    """
    model = load_mode2_model()
    classes = model.classes_
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba([problem_text])[0]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function([problem_text])[0]
        from scipy.special import softmax
        probs = softmax(scores)
    else:
        probs = [1.0 / len(classes)] * len(classes)
    
    # Combine algorithm names with predicted model probabilities
    algo_prob_pairs = []
    for algo_name, prob in zip(classes, probs):
        cat = ALGO_TO_CATEGORY.get(algo_name, "General Algorithm")
        if selected_categories and len(selected_categories) > 0:
            if cat not in selected_categories:
                continue
        algo_prob_pairs.append({
            "algorithm": algo_name,
            "category": cat,
            "ml_probability": round(float(prob) * 100.0, 1)
        })
        
    # Sort descending by ML probability
    algo_prob_pairs.sort(key=lambda x: x["ml_probability"], reverse=True)
    
    # Assign ranks
    results = []
    for idx, item in enumerate(algo_prob_pairs[:k]):
        item["rank"] = idx + 1
        results.append(item)
        
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SuggestAlgo AI — Training Mode 2 Supervised ML Model")
    print("=" * 60)
    
    model, metrics = train_mode2_model()
    
    print("\n--- Training & Test Evaluation Summary ---")
    print(f"Dataset Source: {metrics['dataset_source']}")
    print(f"Train Samples:  {metrics['train_samples']}")
    print(f"Test Samples:   {metrics['test_samples']}")
    print(f"Algorithm Classes: {metrics['num_algorithms']}")
    print(f"Test Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall:    {metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1 Score:  {metrics['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1:     {metrics['f1_weighted'] * 100:.2f}%")
    
    print("\nSample Prediction:")
    sample_text = "I have a list of one million sorted numbers and need to find target in logarithmic time."
    preds = predict_top_k_algorithms(sample_text, selected_categories=["Searching"], k=3)
    for p in preds:
        print(f"  Rank {p['rank']}: {p['algorithm']} ({p['category']}) — ML Model Probability: {p['ml_probability']}%")
