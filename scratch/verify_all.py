import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

print("=" * 70)
print("SUGGESTALGO AI — PHASE 1 FINAL VERIFICATION SUITE")
print("=" * 70)

# 1. KB Counts Verification
from src.algorithm_knowledge_base import ALGORITHM_KNOWLEDGE_BASE
from src.nlp_recommendation import PRIMARY_CATEGORIES, recommend_algorithm_from_text, analyze_natural_language_problem

cat_counts = {c: 0 for c in PRIMARY_CATEGORIES}
for a in ALGORITHM_KNOWLEDGE_BASE:
    cat_counts[a['category']] += 1

print("\n1. KNOWLEDGE BASE CATEGORY COUNTS:")
total_algos = len(ALGORITHM_KNOWLEDGE_BASE)
for c in PRIMARY_CATEGORIES:
    print(f"  - {c:25s}: {cat_counts[c]} algorithms")

assert total_algos >= 100, f"Total algorithms {total_algos} < 100"
assert all(v >= 10 for v in cat_counts.values()), "All categories must have >= 10 algorithms"

# 2. Multi-Category Recommendation Test
print("\n2. TESTING MULTI-CATEGORY RECOMMENDATION:")
prompt_multi = "Find an element in a sorted array and compute shortest path in weighted city graph"
selected_cats = ["Searching", "Graph Algorithms", "Dynamic Programming"]
res_multi = recommend_algorithm_from_text(prompt_multi, selected_categories=selected_cats)

print(f"  Selected categories: {res_multi['selected_categories']}")
for cat, rec in res_multi['category_recommendations'].items():
    print(f"  [{cat}] Top Pick: {rec['top_recommendation']['name']} (Suitability Score: {rec['relevance_score']}%)")
    assert rec['top_recommendation'] is not None

# 3. Text Input Test
print("\n3. TESTING TEXT INPUT MODE 2:")
res_text = recommend_algorithm_from_text("Predict customer churn using tabular usage dataset", selected_categories=["Classification", "Machine Learning / AI"])
assert "Classification" in res_text['category_recommendations']
print("  Text input recommendation successful.")

# 4. Image OCR Test
print("\n4. TESTING OCR GRACEFUL FALLBACK:")
from app import extract_text_from_image
text, success, msg = extract_text_from_image(None)
print(f"  OCR Graceful Fallback Status: success={success}, msg='{msg}'")
assert success == False

# 5. Text + Image Combination Test
print("\n5. TESTING TEXT + IMAGE COMBINATION:")
combined_prompt = "User prompt text\n[Extracted from Image]: Binary search in a sorted array"
res_comb = recommend_algorithm_from_text(combined_prompt, selected_categories=["Searching"])
print(f"  Combined recommendation top pick: {res_comb['category_recommendations']['Searching']['top_recommendation']['name']}")

# 6. Empty Category Selection Test
print("\n6. TESTING EMPTY CATEGORY SELECTION:")
res_empty = recommend_algorithm_from_text("Sort array of numbers", selected_categories=[])
assert len(res_empty['selected_categories']) == 10
print(f"  Empty selection defaulted to all {len(res_empty['selected_categories'])} primary categories.")

# 7. Classification Pipeline Test (Mode 1)
print("\n7. TESTING MODE 1 CLASSIFICATION (IRIS):")
from src.utils import load_sample_dataset
from src.preprocessing import preprocess_dataset
from src.model_evaluation import evaluate_all_models
df_iris, target_iris = load_sample_dataset('iris')
X_iris, y_iris, _, _, _ = preprocess_dataset(df_iris, target_iris)
results_iris, comp_iris, best_iris, _ = evaluate_all_models(X_iris, y_iris, problem_type='classification')
print(f"  Mode 1 Classification Best Model: {best_iris}, Accuracy: {results_iris[best_iris]['accuracy']:.4f}")
assert best_iris is not None

# 8. Regression Pipeline Test (Mode 1)
print("\n8. TESTING MODE 1 REGRESSION (SYNTHETIC):")
rng = np.random.RandomState(42)
X_r = rng.randn(100, 4)
y_r = 2.0 * X_r[:, 0] - 3.0 * X_r[:, 1] + rng.randn(100) * 0.2
df_r = pd.DataFrame(X_r, columns=['f1', 'f2', 'f3', 'f4'])
df_r['target'] = y_r
X_proc, y_proc, _, _, _ = preprocess_dataset(df_r, 'target', problem_type='regression')
results_reg, comp_reg, best_reg, _ = evaluate_all_models(X_proc, y_proc, problem_type='regression')
print(f"  Mode 1 Regression Best Model: {best_reg}, R2 Score: {results_reg[best_reg]['r2']:.4f}")
assert best_reg is not None

print("\n" + "=" * 70)
print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY! ✅")
print("=" * 70)
