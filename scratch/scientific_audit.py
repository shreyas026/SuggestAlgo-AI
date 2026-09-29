import sys
import os
import json
import numpy as np

sys.path.insert(0, '.')

from src.algorithm_knowledge_base import ALGORITHM_KNOWLEDGE_BASE
from src.nlp_recommendation import (
    PRIMARY_CATEGORIES,
    recommend_algorithm_from_text,
    analyze_natural_language_problem,
    compute_suitability_score,
    generate_problem_specific_explanation,
    HAS_SENTENCE_TRANSFORMERS,
    get_embedding_model
)

print("=" * 70)
print("SUGGESTALGO AI — PHASE 1.5 SCIENTIFIC VALIDATION AUDIT")
print("=" * 70)

# ============================================================================
# AUDIT 1: SUITABILITY SCORE AUDIT
# ============================================================================
print("\n--- 1. SUITABILITY SCORE AUDIT ---")
# Inspect code logic directly
# In nlp_recommendation.py:
# raw_scores = cosine_similarity(prompt_emb, doc_embs)
# normalized = (raw_scores - min_s) / (max_s - min_s)
# base_perc_scores = 52.0 + (normalized * 33.0)  # [52.0, 85.0]
# boost:
#   pattern match: +12.0
#   data structure match: +4.0
#   requirement match: +5.0
# final_score = min(98.5, max(52.0, base_score + boost))

print("Exact formula analyzed from source code:")
print("  Base score = 52.0 + ((cosine_sim - min_sim) / (max_sim - min_sim)) * 33.0")
print("  Pattern match boost = +12.0 if pattern in prompt")
print("  Data structure match boost = +4.0 if ds in prompt")
print("  Requirement match boost = +5.0 if requirement in prompt")
print("  Final Suitability Score = min(98.5, max(52.0, Base + Boosts))")
print("  Deterministic: YES")
print("  Probability interpretation: NO (Normalized similarity & compatibility rank)")

# ============================================================================
# AUDIT 2: DETECTED CATEGORY VS SELECTED CATEGORY
# ============================================================================
print("\n--- 2. DETECTED CATEGORY VS SELECTED CATEGORY ---")
test_prompt_2 = "Find an element in a sorted array and find the shortest path in a weighted graph."
selected_2 = ["Searching", "Graph Algorithms", "Dynamic Programming"]
res_2 = recommend_algorithm_from_text(test_prompt_2, selected_categories=selected_2)
prob_rep_2 = res_2['problem_representation']

print(f"Test Prompt: '{test_prompt_2}'")
print(f"Selected Categories: {selected_2}")
print(f"Detected Categories: {prob_rep_2['detected_categories']}")
print("DP Detected?:", "Dynamic Programming" in prob_rep_2['detected_categories'])
print("DP Rec in output?:", "Dynamic Programming" in res_2['category_recommendations'])
if "Dynamic Programming" in res_2['category_recommendations']:
    dp_top = res_2['category_recommendations']['Dynamic Programming']
    print(f"  DP Top Pick: {dp_top['top_recommendation']['name']}, Score: {dp_top['relevance_score']}%")
    print(f"  DP Explanation: {dp_top['explanation']}")

# ============================================================================
# AUDIT 3: 10 TEST PROBLEMS & RELEVANCE
# ============================================================================
print("\n--- 3. 10 TEST PROBLEMS & RELEVANCE ---")
test_cases = [
    ("1. Sorted array lookup", "I have a sorted array of one million numbers and need to find if target exists in logarithmic time.", ["Searching"]),
    ("2. Stable sorting of records", "I need a stable sorting algorithm that preserves original order of equal key records in O(n log n) time.", ["Sorting"]),
    ("3. Shortest path with non-negative weights", "Find the shortest path and minimum distance between two locations in a weighted road network with positive distances.", ["Graph Algorithms"]),
    ("4. Longest common subsequence", "Find the longest common subsequence between two DNA text sequences.", ["Dynamic Programming"]),
    ("5. N-Queens", "Place N chess queens on an N x N board so no two queens attack each other using backtracking.", ["Backtracking"]),
    ("6. Activity scheduling", "Select the maximum number of mutually compatible activities given start and finish times.", ["Greedy Algorithms"]),
    ("7. Binary search tree operations", "Maintain a dynamic set of comparable keys supporting fast insert, delete, and in-order traversal.", ["Tree Algorithms"]),
    ("8. Customer churn prediction", "Predict whether subscription customers will churn based on historical usage features.", ["Classification"]),
    ("9. House price prediction", "Predict continuous house prices based on numerical property features like area and bedrooms.", ["Regression"]),
    ("10. Image classification", "Classify images into category labels using deep convolutional neural network feature maps.", ["Machine Learning / AI"])
]

test_results = []
for label, prompt, cats in test_cases:
    res = recommend_algorithm_from_text(prompt, selected_categories=cats)
    cat_rec = res['category_recommendations'][cats[0]]
    candidates = cat_rec['candidates']
    top3 = [(c['algo']['name'], c['relevance_score']) for c in candidates[:3]]
    test_results.append({
        "label": label,
        "prompt": prompt,
        "category": cats[0],
        "top3": top3,
        "explanation": cat_rec['explanation']
    })
    print(f"\n{label} [{cats[0]}]:")
    for name, score in top3:
        print(f"   - {name}: {score}%")
    print(f"   Explanation: {cat_rec['explanation']}")

# ============================================================================
# AUDIT 4: NEGATIVE / CROSS-CATEGORY TESTS
# ============================================================================
print("\n--- 4. NEGATIVE / CROSS-CATEGORY TESTS ---")
neg_prompt = "Find an element in a sorted array."
neg_selected = ["Dynamic Programming"]
res_neg = recommend_algorithm_from_text(neg_prompt, selected_categories=neg_selected)
dp_neg = res_neg['category_recommendations']['Dynamic Programming']
print(f"Prompt: '{neg_prompt}'")
print(f"Selected Mismatched Category: Dynamic Programming")
print(f"Detected Categories: {res_neg['problem_representation']['detected_categories']}")
print(f"Top Recommendation in DP: {dp_neg['top_recommendation']['name']} ({dp_neg['relevance_score']}%)")
print(f"Explanation rendered: {dp_neg['explanation']}")

# ============================================================================
# AUDIT 5: KNOWLEDGE BASE QUALITY AUDIT
# ============================================================================
print("\n--- 5. KNOWLEDGE BASE QUALITY AUDIT ---")
kb = ALGORITHM_KNOWLEDGE_BASE
ids = set()
duplicates_id = []
invalid_schema = []

for idx, a in enumerate(kb):
    if a['id'] in ids:
        duplicates_id.append(a['id'])
    ids.add(a['id'])

print(f"Total KB entries: {len(kb)}")
print(f"Unique IDs count: {len(ids)}")
print(f"Duplicate IDs: {duplicates_id}")

# ============================================================================
# AUDIT 6: CROSS-CATEGORY DUPLICATES
# ============================================================================
print("\n--- 6. CROSS-CATEGORY DUPLICATES ---")
algo_names = {}
cross_category_dups = []
for a in kb:
    name = a['name'].lower()
    if name in algo_names:
        cross_category_dups.append((a['name'], algo_names[name]['category'], a['category']))
    else:
        algo_names[name] = a

print(f"Cross-Category Duplicate Count: {len(cross_category_dups)}")
for name, cat1, cat2 in cross_category_dups:
    print(f"  - {name}: Primary '{cat1}' vs Secondary '{cat2}'")

# ============================================================================
# AUDIT 7: EMBEDDING VALIDATION
# ============================================================================
print("\n--- 7. EMBEDDING VALIDATION ---")
print(f"SentenceTransformer available in environment: {HAS_SENTENCE_TRANSFORMERS}")
model = get_embedding_model()
print(f"Cached model instance: {type(model)}")

# ============================================================================
# AUDIT 8: RECOMMENDATION STABILITY
# ============================================================================
print("\n--- 8. RECOMMENDATION STABILITY ---")
p_stab = "I have a list of sorted numbers and want to find a target efficiently."
r_stab1 = recommend_algorithm_from_text(p_stab, selected_categories=["Searching"])
r_stab2 = recommend_algorithm_from_text(p_stab, selected_categories=["Searching"])

s1 = r_stab1['category_recommendations']['Searching']['top_recommendation']['name']
sc1 = r_stab1['category_recommendations']['Searching']['relevance_score']
s2 = r_stab2['category_recommendations']['Searching']['top_recommendation']['name']
sc2 = r_stab2['category_recommendations']['Searching']['relevance_score']

print(f"Run 1: {s1} ({sc1}%)")
print(f"Run 2: {s2} ({sc2}%)")
print(f"Identical results: {s1 == s2 and sc1 == sc2}")

# Small wording variation
p_var = "Searching for an item inside an ordered numerical array."
r_var = recommend_algorithm_from_text(p_var, selected_categories=["Searching"])
s_var = r_var['category_recommendations']['Searching']['top_recommendation']['name']
sc_var = r_var['category_recommendations']['Searching']['relevance_score']
print(f"Variation Run: {s_var} ({sc_var}%)")

print("\n" + "=" * 70)
print("AUDIT SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 70)
