"""
SuggestAlgo AI - Natural Language Problem & Project Idea Recommendation Module

Provides semantic ML-based algorithm recommendations for natural-language descriptions of:
1. Computational/Coding Problems (e.g., sorted array search, shortest path)
2. Machine Learning & AI Project Ideas (e.g., customer churn, spam classification, movie recommendation)

Architecture:
User Prompt -> Text Preprocessing -> Semantic Embedding / TF-IDF -> Cosine Similarity vs Knowledge Base -> Ranked Recommendation + Explainability & Complexity Analysis
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.algorithm_knowledge_base import ALGORITHM_KNOWLEDGE_BASE

# Try importing sentence_transformers for deep semantic embeddings, fallback to TF-IDF
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
        " ".join(algo.get('requirements', []))
    ]
    return " ".join(text_parts)


def analyze_natural_language_problem(user_prompt):
    """
    Extract structured problem representation from natural language prompt.
    """
    prompt_lower = user_prompt.lower()
    
    is_ml_project = any(term in prompt_lower for term in [
        'predict', 'classify', 'model', 'churn', 'spam', 'fraud', 'customer',
        'sales', 'house price', 'sentiment', 'image', 'recommendation system',
        'machine learning', 'dataset', 'target', 'movie recommendation'
    ])
    
    is_search = any(term in prompt_lower for term in ['search', 'find', 'lookup', 'exists', 'sorted array', 'index'])
    is_graph = any(term in prompt_lower for term in ['graph', 'path', 'shortest path', 'city', 'nodes', 'edges', 'weighted'])
    is_sorting = any(term in prompt_lower for term in ['sort', 'order', 'arrange', 'rank'])
    
    # Input data structure hints
    data_structure = "Unspecified / General Collection"
    if 'sorted array' in prompt_lower or 'sorted list' in prompt_lower:
        data_structure = "Sorted Array"
    elif 'graph' in prompt_lower or 'cities' in prompt_lower or 'nodes' in prompt_lower:
        data_structure = "Weighted Graph"
    elif 'tree' in prompt_lower:
        data_structure = "Tree / BST"
    elif 'string' in prompt_lower or 'text' in prompt_lower:
        data_structure = "String / Text Sequence"
    elif is_ml_project:
        data_structure = "Tabular Data / Feature Matrix"

    # Category summary
    if is_ml_project:
        if 'movie' in prompt_lower or 'recommend' in prompt_lower:
            category = "Recommendation Systems"
        elif 'sentiment' in prompt_lower or 'spam' in prompt_lower or 'text' in prompt_lower:
            category = "NLP / Text Machine Learning"
        else:
            category = "Machine Learning (Classification / Prediction)"
    elif is_graph:
        category = "Graph Algorithms"
    elif is_search:
        category = "Searching & Hashing"
    elif is_sorting:
        category = "Sorting Algorithms"
    else:
        category = "General Algorithm / Data Structure Strategy"

    word_count = len(user_prompt.split())
    is_vague = word_count < 8 or ("project" in prompt_lower and not any(k in prompt_lower for k in ['size', 'data', 'rows', 'binary', 'multiclass', 'sorted', 'weighted']))

    follow_up_questions = []
    if is_vague or is_ml_project:
        follow_up_questions = [
            "What is the estimated size of your dataset/input data (e.g., 1,000 rows vs 1M elements)?",
            "Is your target variable binary (Yes/No), multiclass, or numerical continuous?",
            "Is the input data pre-sorted, or structured as a graph/tree?",
            "Do you require real-time low-latency inference or maximum predictive accuracy?",
            "Is model interpretability and rule extraction required for your stakeholders?"
        ]

    return {
        "user_prompt": user_prompt,
        "is_ml_project": is_ml_project,
        "category": category,
        "data_structure": data_structure,
        "word_count": word_count,
        "is_vague": is_vague,
        "follow_up_questions": follow_up_questions
    }


def recommend_algorithm_from_text(user_prompt, top_k=4):
    """
    Semantic match user prompt against knowledge base algorithms using embeddings or TF-IDF.
    """
    kb = load_knowledge_base()
    analysis = analyze_natural_language_problem(user_prompt)
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
        perc_scores = 52.0 + (normalized * 43.0)
    else:
        perc_scores = np.full_like(raw_scores, 75.0)

    # Specific heuristic boosts for explicit problem patterns
    p_lower = user_prompt.lower()
    for i, algo in enumerate(kb):
        p_patterns = algo.get('problem_patterns', [])
        for pattern in p_patterns:
            if pattern.lower() in p_lower:
                perc_scores[i] = min(98.5, perc_scores[i] + 18.0)
                break

    sorted_indices = np.argsort(perc_scores)[::-1]
    
    ranked_candidates = []
    for idx in sorted_indices[:top_k]:
        algo = kb[idx]
        score = round(float(perc_scores[idx]), 1)
        ranked_candidates.append({
            "algo": algo,
            "similarity_score": score
        })

    top_algo = ranked_candidates[0]['algo'] if ranked_candidates else None
    top_score = ranked_candidates[0]['similarity_score'] if ranked_candidates else 0.0

    return {
        "analysis": analysis,
        "top_recommendation": top_algo,
        "top_score": top_score,
        "candidates": ranked_candidates,
        "all_scores": perc_scores
    }
