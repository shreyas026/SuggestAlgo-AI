import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

# Paths
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

full_csv_path = os.path.join(data_dir, 'problem_algorithm_dataset.csv')
df = pd.read_csv(full_csv_path)

print(f"Loaded dataset: {len(df)} total rows across {df['algorithm'].nunique()} algorithms.")

# ---------------------------------------------------------
# PHASE 2 & 3: 70% Train, 15% Val, 15% Test Stratified Split
# ---------------------------------------------------------
# First split: 70% train, 30% temp (val + test)
df_train, df_temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df['algorithm']
)

# Second split: split temp into 50% val (15% total) and 50% test (15% total)
df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,
    random_state=42,
    stratify=df_temp['algorithm']
)

train_path = os.path.join(data_dir, 'train_algorithm_dataset.csv')
val_path = os.path.join(data_dir, 'validation_algorithm_dataset.csv')
test_path = os.path.join(data_dir, 'test_algorithm_dataset.csv')

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)
df_test.to_csv(test_path, index=False)

# Backward compatibility copies
df_train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
df_test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

print(f"Dataset split complete: Train={len(df_train)} (70%), Val={len(df_val)} (15%), Test={len(df_test)} (15%).")

# ---------------------------------------------------------
# PHASE 9: Leakage Audit
# ---------------------------------------------------------
train_texts = set(df_train['problem_description'])
val_texts = set(df_val['problem_description'])
test_texts = set(df_test['problem_description'])

overlap_tv = train_texts.intersection(val_texts)
overlap_tt = train_texts.intersection(test_texts)
overlap_vt = val_texts.intersection(test_texts)

print(f"Leakage Audit:")
print(f" - Exact overlap between Train & Val: {len(overlap_tv)}")
print(f" - Exact overlap between Train & Test: {len(overlap_tt)}")
print(f" - Exact overlap between Val & Test: {len(overlap_vt)}")
assert len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0, "Leakage detected!"

# ---------------------------------------------------------
# PHASE 5: Model Experiments on Validation Set
# ---------------------------------------------------------
# TF-IDF Vectorizer fit strictly on TRAIN dataset ONLY
tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
X_train_tfidf = tfidf.fit_transform(df_train['problem_description'])
X_val_tfidf = tfidf.transform(df_val['problem_description'])
X_test_tfidf = tfidf.transform(df_test['problem_description'])

y_train = df_train['algorithm']
y_val = df_val['algorithm']
y_test = df_test['algorithm']

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42, max_iter=3000),
    "Naive Bayes": MultinomialNB(),
    "SGD Classifier": SGDClassifier(loss='log_loss', max_iter=3000, random_state=42)
}

comparison_results = []
trained_pipelines = {}

print("\n--- Training Model Candidates & Evaluating on Validation Set ---")
for name, clf in models.items():
    start_t = time.time()
    clf.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_t
    
    val_preds = clf.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, val_preds)
    val_macro_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    val_weighted_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
    
    comparison_results.append({
        "Model": name,
        "Validation Accuracy": round(val_acc, 4),
        "Validation Macro F1": round(val_macro_f1, 4),
        "Validation Weighted F1": round(val_weighted_f1, 4),
        "Training Time (s)": round(train_time, 4)
    })
    
    # Store complete pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', clf)
    ])
    trained_pipelines[name] = pipeline
    print(f" -> {name}: Val Acc = {val_acc:.4f} | Val Macro F1 = {val_macro_f1:.4f}")

df_comp = pd.DataFrame(comparison_results)
comp_csv_path = os.path.join(results_dir, 'model_comparison.csv')
df_comp.to_csv(comp_csv_path, index=False)
print(f"Saved model comparison to {comp_csv_path}.")

# Select best model based on Validation Macro F1
best_model_name = df_comp.sort_values(by="Validation Macro F1", ascending=False).iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
print(f"\nBest Selected Model based on Validation: {best_model_name}")

# ---------------------------------------------------------
# PHASE 6: Hierarchical Recommendation Experiment
# ---------------------------------------------------------
# Level 1: Predict Category from Problem
y_train_cat = df_train['category']
y_val_cat = df_val['category']
y_test_cat = df_test['category']

cat_clf = LogisticRegression(max_iter=3000, random_state=42)
cat_clf.fit(X_train_tfidf, y_train_cat)
val_cat_preds = cat_clf.predict(X_val_tfidf)
test_cat_preds = cat_clf.predict(X_test_tfidf)
val_cat_acc = accuracy_score(y_val_cat, val_cat_preds)
test_cat_acc = accuracy_score(y_test_cat, test_cat_preds)
print(f"Hierarchical Level 1 (Category Classification) Test Accuracy: {test_cat_acc:.4f}")

# ---------------------------------------------------------
# PHASE 8: Final Untouched Test Set Evaluation for Selected Model
# ---------------------------------------------------------
test_preds = best_pipeline.predict(df_test['problem_description'])
test_proba = best_pipeline.predict_proba(df_test['problem_description']) if hasattr(best_pipeline.named_steps['classifier'], 'predict_proba') else None

# Calculate Top-K Accuracy
classes = list(best_pipeline.named_steps['classifier'].classes_)
def compute_top_k_accuracy(proba_matrix, y_true, classes_list, k):
    correct = 0
    for i, true_label in enumerate(y_true):
        top_k_indices = np.argsort(proba_matrix[i])[::-1][:k]
        top_k_classes = [classes_list[idx] for idx in top_k_indices]
        if true_label in top_k_classes:
            correct += 1
    return correct / len(y_true)

top1_acc = accuracy_score(y_test, test_preds)
top3_acc = compute_top_k_accuracy(test_proba, y_test, classes, 3) if test_proba is not None else top1_acc
top5_acc = compute_top_k_accuracy(test_proba, y_test, classes, 5) if test_proba is not None else top1_acc

prec_macro = precision_score(y_test, test_preds, average='macro', zero_division=0)
prec_weighted = precision_score(y_test, test_preds, average='weighted', zero_division=0)
rec_macro = recall_score(y_test, test_preds, average='macro', zero_division=0)
rec_weighted = recall_score(y_test, test_preds, average='weighted', zero_division=0)
f1_mac = f1_score(y_test, test_preds, average='macro', zero_division=0)
f1_weight = f1_score(y_test, test_preds, average='weighted', zero_division=0)

final_results = {
    "dataset_type": "Independent Algorithm Recommendation Benchmark",
    "dataset_source": "SuggestAlgo AI Knowledge Base (Curated Synthetic Benchmark)",
    "total_samples": len(df),
    "train_samples": len(df_train),
    "validation_samples": len(df_val),
    "test_samples": len(df_test),
    "num_algorithms": len(classes),
    "selected_model": best_model_name,
    "vectorizer": "TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)",
    "accuracy": round(top1_acc, 4),
    "top_3_accuracy": round(top3_acc, 4),
    "top_5_accuracy": round(top5_acc, 4),
    "precision_macro": round(prec_macro, 4),
    "precision_weighted": round(prec_weighted, 4),
    "recall_macro": round(rec_macro, 4),
    "recall_weighted": round(rec_weighted, 4),
    "f1_macro": round(f1_mac, 4),
    "f1_weighted": round(f1_weight, 4),
    "hierarchical_category_accuracy": round(test_cat_acc, 4),
    "random_state": 42,
    "leakage_verified": True
}

final_json_path = os.path.join(results_dir, 'final_test_results.json')
with open(final_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2)
print(f"Saved final untouched test results to {final_json_path}.")

# Also save mode2_ml_results.json for backward compatibility
with open(os.path.join(results_dir, 'mode2_ml_results.json'), 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2)

# Save confusion matrix plot
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title(f'Confusion Matrix - {best_model_name} (112 Classes)', fontsize=14)
plt.xlabel('Predicted Algorithm')
plt.ylabel('Actual Algorithm')
plt.tight_layout()
cm_path = os.path.join(results_dir, 'mode2_confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved confusion matrix plot to {cm_path}.")

# ---------------------------------------------------------
# PHASE 11: Save Model Artifacts
# ---------------------------------------------------------
best_model_path = os.path.join(models_dir, 'mode2_algorithm_classifier.joblib')
joblib.dump(best_pipeline, best_model_path)
print(f"Saved final trained model pipeline to {best_model_path}.")

print("\n=== EXPERIMENTATION COMPLETE ===")
print(json.dumps(final_results, indent=2))
