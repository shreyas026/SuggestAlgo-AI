"""
SuggestAlgo AI - Natural Language Problem & Category-Aware Recommendation Module

Provides semantic ML-based algorithm recommendations for natural-language descriptions of:
1. Computational/Coding Problems (e.g., sorted array search, shortest path, knapsack)
2. Machine Learning & AI Project Ideas (e.g., customer churn, spam classification, movie recommendation)

Architecture:
User Prompt -> Problem Representation Extraction -> Category Filtering -> Semantic Embedding / TF-IDF
           -> Transparent Suitability Scoring -> Category-Specific Recommendation & Explainability
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.algorithm_knowledge_base import ALGORITHM_KNOWLEDGE_BASE

# Standardized 10 Primary Categories
PRIMARY_CATEGORIES = [
    "Searching",
    "Sorting",
    "Classification",
    "Regression",
    "Graph Algorithms",
    "Tree Algorithms",
    "Dynamic Programming",
    "Greedy Algorithms",
    "Backtracking",
    "Machine Learning / AI"
]

# Legacy specialized category mapping for backwards compatibility
CATEGORY_MAPPING = {
    "Hashing": "Searching",
    "Recommendation Systems": "Machine Learning / AI",
    "NLP / Text Machine Learning": "Machine Learning / AI",
    "Optimization": "Dynamic Programming",
    "Two Pointers": "Searching",
    "Sliding Window": "Searching"
}

HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False


_EMBEDDING_MODEL_CACHE = None


def load_knowledge_base():
    """Return algorithm knowledge base."""
    return ALGORITHM_KNOWLEDGE_BASE


def get_embedding_model():
    """Load or return cached sentence transformer model if available."""
    global _EMBEDDING_MODEL_CACHE
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    
    if _EMBEDDING_MODEL_CACHE is None:
        try:
            _EMBEDDING_MODEL_CACHE = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _EMBEDDING_MODEL_CACHE = None
            
    return _EMBEDDING_MODEL_CACHE


def _create_algorithm_document(algo):
    """Combine algorithm fields into a comprehensive text document for embedding/vectorizing."""
    text_parts = [
        algo.get('name', ''),
        algo.get('category', ''),
        algo.get('description', ''),
        " ".join(algo.get('problem_patterns', [])),
        " ".join(algo.get('data_structures', [])),
        " ".join(algo.get('best_use_cases', [])),
        " ".join(algo.get('requirements', [])),
        " ".join(algo.get('advantages', []))
    ]
    return " ".join(text_parts)


def analyze_natural_language_problem(user_prompt):
    """
    Extract a structured internal problem representation from a natural language prompt.
    Returns structured representation containing problem description, detected categories,
    data structures, input characteristics, constraints, input size, optimization goal,
    required output, and detected patterns.
    """
    prompt_lower = user_prompt.lower()
    
    # 1. Detect ML project vs Classical Algorithm
    is_ml_project = any(term in prompt_lower for term in [
        'predict', 'classify', 'model', 'churn', 'spam', 'fraud', 'customer',
        'sales', 'house price', 'sentiment', 'image', 'recommendation system',
        'machine learning', 'dataset', 'target', 'movie recommendation', 'clustering', 'forecast'
    ])
    
    # 2. Detect Categories
    detected_cats = []
    if any(k in prompt_lower for k in ['search', 'find', 'lookup', 'exists', 'binary search', 'index', 'hash', 'two sum']):
        detected_cats.append("Searching")
    if any(k in prompt_lower for k in ['sort', 'order', 'arrange', 'rank', 'alphabetical']):
        detected_cats.append("Sorting")
    if any(k in prompt_lower for k in ['classify', 'classification', 'spam', 'churn', 'fraud', 'label']):
        detected_cats.append("Classification")
    if any(k in prompt_lower for k in ['regression', 'predict price', 'continuous target', 'house price', 'forecast sales']):
        detected_cats.append("Regression")
    if any(k in prompt_lower for k in ['graph', 'shortest path', 'dijkstra', 'city', 'nodes', 'edges', 'connected components', 'topological']):
        detected_cats.append("Graph Algorithms")
    if any(k in prompt_lower for k in ['tree', 'bst', 'binary tree', 'trie', 'autocomplete', 'heap', 'prefix tree']):
        detected_cats.append("Tree Algorithms")
    if any(k in prompt_lower for k in ['knapsack', 'dynamic programming', 'dp', 'lcs', 'subsequence', 'coin change', 'edit distance']):
        detected_cats.append("Dynamic Programming")
    if any(k in prompt_lower for k in ['greedy', 'activity selection', 'huffman', 'interval scheduling', 'fractional']):
        detected_cats.append("Greedy Algorithms")
    if any(k in prompt_lower for k in ['backtracking', 'n-queens', 'sudoku', 'maze', 'permutation', 'combination', 'subset']):
        detected_cats.append("Backtracking")
    if is_ml_project or any(k in prompt_lower for k in ['kmeans', 'dbscan', 'pca', 'neural network', 'cnn', 'lstm', 'transformer', 'recommend']):
        detected_cats.append("Machine Learning / AI")

    if not detected_cats:
        detected_cats = ["not specified"]

    # 3. Detect Data Structures
    detected_ds = []
    if 'sorted array' in prompt_lower or 'sorted list' in prompt_lower:
        detected_ds.append("Sorted Array")
    elif 'array' in prompt_lower or 'list' in prompt_lower:
        detected_ds.append("Array")
    if 'graph' in prompt_lower or 'cities' in prompt_lower or 'network' in prompt_lower:
        detected_ds.append("Weighted / Directed Graph")
    if 'tree' in prompt_lower or 'bst' in prompt_lower:
        detected_ds.append("Tree Structure")
    if 'grid' in prompt_lower or 'matrix' in prompt_lower or 'board' in prompt_lower or 'maze' in prompt_lower:
        detected_ds.append("2D Grid / Matrix")
    if 'string' in prompt_lower or 'text' in prompt_lower or 'word' in prompt_lower:
        detected_ds.append("String / Text Sequence")
    if 'hash' in prompt_lower or 'dictionary' in prompt_lower or 'map' in prompt_lower:
        detected_ds.append("Hash Map / Dictionary")
    if is_ml_project:
        detected_ds.append("Tabular Data / Feature Matrix")
    if not detected_ds:
        detected_ds = ["unknown"]

    # 4. Detect Input Characteristics
    input_chars = []
    if 'sorted' in prompt_lower: input_chars.append("Pre-sorted input")
    if 'unsorted' in prompt_lower: input_chars.append("Unsorted collection")
    if 'weighted' in prompt_lower: input_chars.append("Weighted edges")
    if 'unweighted' in prompt_lower: input_chars.append("Unweighted graph")
    if 'integer' in prompt_lower or 'numbers' in prompt_lower: input_chars.append("Numeric sequence")
    if 'text' in prompt_lower or 'document' in prompt_lower: input_chars.append("Text sequence")
    if not input_chars:
        input_chars = ["not specified"]

    # 5. Detect Constraints
    constraints = []
    if 'logarithmic' in prompt_lower or 'o(log n)' in prompt_lower: constraints.append("Logarithmic time constraint O(log n)")
    if 'linear' in prompt_lower or 'o(n)' in prompt_lower: constraints.append("Linear time constraint O(n)")
    if 'in-place' in prompt_lower or 'no extra memory' in prompt_lower: constraints.append("In-place memory constraint O(1)")
    if 'real-time' in prompt_lower or 'low-latency' in prompt_lower: constraints.append("Real-time low latency requirement")
    if 'interpretable' in prompt_lower or 'explainable' in prompt_lower: constraints.append("Interpretability constraint")
    if not constraints:
        constraints = ["not specified"]

    # 6. Detect Input Size
    input_size = "unknown"
    if any(k in prompt_lower for k in ['million', '1m', '10^6', '1000000']):
        input_size = "Large (~1,000,000 elements)"
    elif any(k in prompt_lower for k in ['thousand', '1k', '1000']):
        input_size = "Medium (~1,000 elements)"
    elif any(k in prompt_lower for k in ['small', 'few', 'tiny']):
        input_size = "Small (<100 elements)"

    # 7. Detect Optimization Goal
    optimization_goal = "not specified"
    if 'shortest path' in prompt_lower or 'shortest distance' in prompt_lower:
        optimization_goal = "Minimize path distance / edge weight"
    elif 'fast' in prompt_lower or 'efficient' in prompt_lower or 'lookup' in prompt_lower:
        optimization_goal = "Minimize search time complexity"
    elif 'sort' in prompt_lower:
        optimization_goal = "Order elements monotonically"
    elif 'maximum profit' in prompt_lower or 'maximize' in prompt_lower:
        optimization_goal = "Maximize total value / profit"
    elif 'predict' in prompt_lower or 'classify' in prompt_lower:
        optimization_goal = "Maximize predictive accuracy / F1 score"

    # 8. Required Output
    required_output = "not specified"
    if 'find index' in prompt_lower or 'element exists' in prompt_lower or 'lookup' in prompt_lower:
        required_output = "Index position or boolean existence flag"
    elif 'path' in prompt_lower or 'sequence' in prompt_lower:
        required_output = "Optimal path / node sequence"
    elif 'predict' in prompt_lower or 'churn' in prompt_lower:
        required_output = "Target class prediction / probability score"
    elif 'clusters' in prompt_lower or 'segment' in prompt_lower:
        required_output = "Cluster group assignments"

    # 9. Detected Problem Patterns
    detected_patterns = []
    kb = load_knowledge_base()
    for algo in kb:
        for pat in algo.get('problem_patterns', []):
            if pat.lower() in prompt_lower:
                if pat not in detected_patterns:
                    detected_patterns.append(pat)

    if not detected_patterns:
        detected_patterns = ["general query matching"]

    word_count = len(user_prompt.split())
    is_vague = word_count < 7 or ("project" in prompt_lower and not any(k in prompt_lower for k in ['size', 'data', 'rows', 'binary', 'multiclass', 'sorted', 'weighted']))

    follow_up_questions = []
    if is_vague or is_ml_project:
        follow_up_questions = [
            "What is the estimated size of your dataset/input data (e.g., 1,000 rows vs 1M elements)?",
            "Is your target variable binary (Yes/No), multiclass, or numerical continuous?",
            "Is the input data pre-sorted, or structured as a graph/tree?",
            "Do you require real-time low-latency inference or maximum predictive accuracy?",
            "Is model interpretability and rule extraction required for your stakeholders?"
        ]

    primary_cat = detected_cats[0] if detected_cats and detected_cats[0] != "not specified" else "General Algorithm Strategy"
    primary_ds = detected_ds[0] if detected_ds and detected_ds[0] != "unknown" else "Unspecified Data Structure"

    return {
        "problem_description": user_prompt,
        "detected_categories": detected_cats,
        "data_structures": detected_ds,
        "input_characteristics": input_chars,
        "constraints": constraints,
        "input_size": input_size,
        "optimization_goal": optimization_goal,
        "required_output": required_output,
        "detected_patterns": detected_patterns,
        # Backwards compatibility fields
        "user_prompt": user_prompt,
        "is_ml_project": is_ml_project,
        "category": primary_cat,
        "data_structure": primary_ds,
        "word_count": word_count,
        "is_vague": is_vague,
        "follow_up_questions": follow_up_questions
    }


def compute_suitability_score(algo, user_prompt, base_similarity_score):
    """
    Transparent Suitability Score Calculation:
    Score = Normalized Semantic Match (50-85%) + Characteristic/Requirements Compatibility Boost (0-15%)
    Resulting score is bounded between 52.0% and 98.5%.
    """
    p_lower = user_prompt.lower()
    score = float(base_similarity_score)
    
    boost = 0.0
    # Match problem patterns
    for pat in algo.get('problem_patterns', []):
        if pat.lower() in p_lower:
            boost += 12.0
            break
            
    # Match data structures
    for ds in algo.get('data_structures', []):
        if ds.lower() in p_lower:
            boost += 4.0
            break
            
    # Match requirements / characteristics
    for req in algo.get('requirements', []):
        if 'sorted' in req.lower() and 'sorted' in p_lower:
            boost += 5.0
            break
            
    final_score = min(98.5, max(52.0, score + boost))
    return round(final_score, 1)


def generate_problem_specific_explanation(problem_rep, algo):
    """
    Produce a problem-specific explanation by combining the structured problem representation
    and algorithm metadata.
    """
    desc = problem_rep['problem_description']
    ds = problem_rep['data_structures'][0] if problem_rep['data_structures'] else "input collection"
    goal = problem_rep['optimization_goal']
    
    explanation_parts = [
        f"The problem specifies: '{desc}'.",
    ]
    
    if ds != "unknown":
        explanation_parts.append(f"It operates on a `{ds}` structure.")
    if goal != "not specified":
        explanation_parts.append(f"The objective is to {goal.lower()}.")
        
    explanation_parts.append(
        f"**{algo['name']}** is recommended for the **{algo['category']}** category because {algo['description'].lower()}"
    )
    
    if algo.get('best_use_cases'):
        use_case = algo['best_use_cases'][0]
        explanation_parts.append(f"It is especially well-suited for {use_case.lower()}.")
        
    if algo.get('advantages'):
        adv = algo['advantages'][0]
        explanation_parts.append(f"Key advantage: {adv}.")
        
    return " ".join(explanation_parts)


def recommend_algorithm_from_text(user_prompt, selected_categories=None, top_k=4):
    """
    Semantic match user prompt against knowledge base algorithms for user-selected categories.
    
    Parameters:
    - user_prompt (str): Natural language description of problem or project idea.
    - selected_categories (list or None): List of selected category strings. If None or empty, defaults to all 10 PRIMARY_CATEGORIES.
    - top_k (int): Number of top candidates to return per category.
    
    Returns:
    - Dict with problem_representation, category_recommendations, and overall top_recommendation for backwards compatibility.
    """
    kb = load_knowledge_base()
    analysis = analyze_natural_language_problem(user_prompt)
    
    if not selected_categories:
        selected_categories = list(PRIMARY_CATEGORIES)
    else:
        # Normalize category names if legacy names are passed
        selected_categories = [CATEGORY_MAPPING.get(c, c) for c in selected_categories]
        selected_categories = [c for c in selected_categories if c in PRIMARY_CATEGORIES]
        if not selected_categories:
            selected_categories = list(PRIMARY_CATEGORIES)

    # Pre-compute semantic similarity vectors across full KB
    documents = [_create_algorithm_document(algo) for algo in kb]
    model = get_embedding_model()
    
    if model is not None:
        doc_embeddings = model.encode(documents, convert_to_numpy=True)
        prompt_embedding = model.encode([user_prompt], convert_to_numpy=True)
        similarities = cosine_similarity(prompt_embedding, doc_embeddings)[0]
    else:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(documents + [user_prompt])
        prompt_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        similarities = cosine_similarity(prompt_vec, doc_vecs)[0]

    raw_scores = similarities
    max_s = np.max(raw_scores) if len(raw_scores) > 0 else 1.0
    min_s = np.min(raw_scores) if len(raw_scores) > 0 else 0.0
    
    if max_s > min_s:
        normalized = (raw_scores - min_s) / (max_s - min_s)
        base_perc_scores = 52.0 + (normalized * 33.0)  # Base 52-85%
    else:
        base_perc_scores = np.full_like(raw_scores, 70.0)

    # Compute final suitability scores per KB entry
    final_scores = []
    for i, algo in enumerate(kb):
        score = compute_suitability_score(algo, user_prompt, base_perc_scores[i])
        final_scores.append(score)

    # Category-Specific Recommendation Partitioning
    category_recommendations = {}
    all_category_top_candidates = []

    for category in selected_categories:
        cat_indices = [i for i, algo in enumerate(kb) if algo['category'] == category]
        if not cat_indices:
            continue
            
        cat_scores = [final_scores[i] for i in cat_indices]
        sorted_cat_order = np.argsort(cat_scores)[::-1]
        
        ranked_cat_candidates = []
        for idx_in_order in sorted_cat_order[:top_k]:
            orig_idx = cat_indices[idx_in_order]
            algo = kb[orig_idx]
            score = final_scores[orig_idx]
            ranked_cat_candidates.append({
                "algo": algo,
                "similarity_score": score,
                "relevance_score": score
            })
            
        top_cat_algo = ranked_cat_candidates[0]['algo']
        top_cat_score = ranked_cat_candidates[0]['relevance_score']
        explanation = generate_problem_specific_explanation(analysis, top_cat_algo)
        
        category_recommendations[category] = {
            "category": category,
            "top_recommendation": top_cat_algo,
            "relevance_score": top_cat_score,
            "explanation": explanation,
            "candidates": ranked_cat_candidates,
            "detected_characteristics": analysis['input_characteristics'],
            "requirements": top_cat_algo.get('requirements', []),
            "time_complexity": top_cat_algo.get('time_complexity', 'O(N)'),
            "space_complexity": top_cat_algo.get('space_complexity', 'O(1)'),
            "advantages": top_cat_algo.get('advantages', []),
            "limitations": top_cat_algo.get('limitations', []),
            "alternatives": top_cat_algo.get('alternatives', [])
        }
        
        all_category_top_candidates.append({
            "algo": top_cat_algo,
            "similarity_score": top_cat_score,
            "category": category
        })

    # Overall top recommendation across selected categories (for backwards compatibility)
    all_category_top_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
    global_top_algo = all_category_top_candidates[0]['algo'] if all_category_top_candidates else kb[0]
    global_top_score = all_category_top_candidates[0]['similarity_score'] if all_category_top_candidates else 75.0

    return {
        "analysis": analysis,
        "problem_representation": analysis,
        "selected_categories": selected_categories,
        "category_recommendations": category_recommendations,
        "top_recommendation": global_top_algo,
        "top_score": global_top_score,
        "candidates": all_category_top_candidates,
        "all_scores": final_scores
    }
