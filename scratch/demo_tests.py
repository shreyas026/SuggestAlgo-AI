import sys
sys.path.insert(0, '.')

from src.nlp_recommendation import recommend_algorithm_from_text, PRIMARY_CATEGORIES
from src.mode2_ml_model import predict_top_k_algorithms

demo_prompts = [
    ("TEST 1: Sorted Array Search", "Find an element in a sorted array of one million numbers.", ["Searching"]),
    ("TEST 2: Shortest Path Weighted Graph", "Find the shortest path between nodes in a weighted graph with non-negative edge weights.", ["Graph Algorithms"]),
    ("TEST 3: Longest Common Subsequence", "Find the longest common subsequence between two strings.", ["Dynamic Programming"]),
    ("TEST 4: N-Queens Puzzle", "Solve the N Queens puzzle.", ["Backtracking"]),
    ("TEST 5: Activity Selection", "Select the maximum number of non-overlapping activities.", ["Greedy Algorithms"]),
    ("TEST 6: Customer Churn ML", "I want to predict whether customers will leave a subscription service.", ["Classification", "Machine Learning / AI"])
]

print("=" * 70)
print("SUGGESTALGO AI — DEMO TEST CASES VALIDATION (HYBRID MODE 2)")
print("=" * 70)

for label, prompt, cats in demo_prompts:
    print(f"\n{label}:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Selected Categories: {cats}")
    
    # Semantic Retrieval
    rec_data = recommend_algorithm_from_text(prompt, selected_categories=cats)
    cat_recs = rec_data['category_recommendations']
    
    # Supervised ML Model
    ml_preds = predict_top_k_algorithms(prompt, selected_categories=cats, k=2)
    
    for c in cats:
        if c in cat_recs:
            sem_top = cat_recs[c]['top_recommendation']['name']
            sem_score = cat_recs[c]['relevance_score']
            print(f"    [{c}] Semantic Retrieval Pick: {sem_top} (Suitability Score: {sem_score}%)")
            
    if ml_preds:
        top_ml = ml_preds[0]
        print(f"    Supervised ML Model Rank 1: {top_ml['algorithm']} ({top_ml['category']}) — ML Model Probability: {top_ml['ml_probability']}%")

print("\n" + "=" * 70)
print("DEMO TEST CASES VALIDATION COMPLETED SUCCESSFULLY! ✅")
print("=" * 70)
