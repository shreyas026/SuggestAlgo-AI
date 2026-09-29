"""
SuggestAlgo AI - Mode 2 Category-Aware Recommendation Test Suite

Tests Phase 1 Mode 2 requirements:
1. Category list validation (10 primary categories)
2. Knowledge base schema completeness (all 16 required fields)
3. Minimum algorithm coverage (10-15 algorithms per category)
4. Category filtering
5. Multi-category recommendation
6. Empty category selection
7. Text-only Mode 2
8. Image/OCR Mode 2 (graceful fallback)
9. Text + Image Mode 2
10. Ranking score calculation & transparency
11. Problem-specific explanation generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.nlp_recommendation import (
    PRIMARY_CATEGORIES,
    ALGORITHM_KNOWLEDGE_BASE,
    analyze_natural_language_problem,
    recommend_algorithm_from_text,
    compute_suitability_score,
    generate_problem_specific_explanation
)
from app import extract_text_from_image

REQUIRED_SCHEMA_FIELDS = [
    "id", "name", "category", "description", "problem_patterns",
    "data_structures", "requirements", "input_characteristics",
    "constraints", "best_use_cases", "time_complexity", "space_complexity",
    "advantages", "limitations", "alternatives", "python_template"
]


def test_1_category_list_validation():
    """Verify exact 10 primary categories exist."""
    print("Testing 1. Category list validation...", end=" ")
    expected = [
        "Searching", "Sorting", "Classification", "Regression",
        "Graph Algorithms", "Tree Algorithms", "Dynamic Programming",
        "Greedy Algorithms", "Backtracking", "Machine Learning / AI"
    ]
    assert PRIMARY_CATEGORIES == expected, f"Categories mismatch. Expected {expected}, got {PRIMARY_CATEGORIES}"
    assert len(PRIMARY_CATEGORIES) == 10
    print("PASSED ✅")


def test_2_knowledge_base_schema():
    """Verify every algorithm entry contains all 16 required schema fields."""
    print("Testing 2. Knowledge base schema...", end=" ")
    kb = ALGORITHM_KNOWLEDGE_BASE
    assert len(kb) >= 100, f"Expected at least 100 algorithms, got {len(kb)}"
    
    for idx, algo in enumerate(kb):
        for field in REQUIRED_SCHEMA_FIELDS:
            assert field in algo, f"Algorithm at index {idx} ({algo.get('id', 'unknown')}) missing field '{field}'"
            val = algo[field]
            if isinstance(val, (str, list)):
                assert len(val) > 0, f"Field '{field}' in '{algo['id']}' is empty"
    print("PASSED ✅")


def test_3_minimum_algorithm_coverage():
    """Verify 10-15 algorithms exist per primary category."""
    print("Testing 3. Minimum algorithm coverage per category...", end=" ")
    kb = ALGORITHM_KNOWLEDGE_BASE
    cat_counts = {cat: 0 for cat in PRIMARY_CATEGORIES}
    
    for algo in kb:
        cat = algo['category']
        assert cat in cat_counts, f"Unknown category '{cat}' found in algorithm '{algo['id']}'"
        cat_counts[cat] += 1
        
    for cat, count in cat_counts.items():
        assert count >= 10, f"Category '{cat}' has only {count} algorithms, expected at least 10"
        assert count <= 15, f"Category '{cat}' has {count} algorithms, expected at most 15"
        
    print("PASSED ✅ (" + ", ".join([f"{c}: {n}" for c, n in cat_counts.items()]) + ")")


def test_4_category_filtering():
    """Verify recommendations are strictly filtered by selected categories."""
    print("Testing 4. Category filtering...", end=" ")
    prompt = "I need to find an element in a sorted list of numbers"
    
    # Select only Searching and Dynamic Programming
    selected = ["Searching", "Dynamic Programming"]
    res = recommend_algorithm_from_text(prompt, selected_categories=selected)
    
    rec_cats = list(res['category_recommendations'].keys())
    assert set(rec_cats) == set(selected), f"Expected categories {selected}, got {rec_cats}"
    assert "Sorting" not in rec_cats
    assert "Classification" not in rec_cats
    print("PASSED ✅")


def test_5_multi_category_recommendation():
    """Verify multi-category recommendation returns top pick for each selected category."""
    print("Testing 5. Multi-category recommendation...", end=" ")
    prompt = "Predict customer churn and find shortest path for delivery network"
    selected = ["Classification", "Graph Algorithms", "Machine Learning / AI"]
    
    res = recommend_algorithm_from_text(prompt, selected_categories=selected)
    recs = res['category_recommendations']
    
    assert "Classification" in recs
    assert "Graph Algorithms" in recs
    assert "Machine Learning / AI" in recs
    
    assert recs["Classification"]["top_recommendation"] is not None
    assert recs["Graph Algorithms"]["top_recommendation"] is not None
    assert recs["Machine Learning / AI"]["top_recommendation"] is not None
    
    assert "Dijkstra" in recs["Graph Algorithms"]["top_recommendation"]["name"] or "BFS" in recs["Graph Algorithms"]["top_recommendation"]["name"] or "Graph" in recs["Graph Algorithms"]["top_recommendation"]["category"]
    print("PASSED ✅")


def test_6_empty_category_selection():
    """Verify empty category selection defaults to all 10 primary categories."""
    print("Testing 6. Empty category selection...", end=" ")
    prompt = "Sort a large array of numbers"
    
    # Pass empty list
    res_empty = recommend_algorithm_from_text(prompt, selected_categories=[])
    assert len(res_empty['selected_categories']) == 10
    assert len(res_empty['category_recommendations']) == 10
    
    # Pass None
    res_none = recommend_algorithm_from_text(prompt, selected_categories=None)
    assert len(res_none['selected_categories']) == 10
    print("PASSED ✅")


def test_7_text_only_mode2():
    """Test text-only problem recommendation."""
    print("Testing 7. Text-only Mode 2...", end=" ")
    res = recommend_algorithm_from_text("Find 0/1 knapsack maximum profit for items with weights and values")
    
    dp_rec = res['category_recommendations'].get('Dynamic Programming')
    assert dp_rec is not None
    assert "Knapsack" in dp_rec['top_recommendation']['name']
    assert dp_rec['relevance_score'] > 60.0
    print("PASSED ✅")


def test_8_image_ocr_mode2():
    """Test image OCR graceful fallback when no image or missing tesseract."""
    print("Testing 8. Image/OCR Mode 2 graceful handling...", end=" ")
    text, success, msg = extract_text_from_image(None)
    assert success == False
    assert isinstance(msg, str)
    assert len(msg) > 0
    print(f"PASSED ✅ (Status: '{msg}')")


def test_9_text_plus_image_mode2():
    """Test combined text and OCR prompt analysis."""
    print("Testing 9. Text + Image Mode 2...", end=" ")
    ocr_text = "I have a graph with weighted edges and need shortest path"
    user_text = "User adds: graph has 500 nodes"
    combined = f"{user_text}\n[Extracted from Image]: {ocr_text}"
    
    res = recommend_algorithm_from_text(combined, selected_categories=["Graph Algorithms"])
    graph_rec = res['category_recommendations']['Graph Algorithms']
    assert graph_rec['top_recommendation'] is not None
    assert "Graph" in graph_rec['top_recommendation']['category']
    print("PASSED ✅")


def test_10_ranking_score_calculation():
    """Test ranking score transparency and bounds."""
    print("Testing 10. Ranking score calculation & transparency...", end=" ")
    res = recommend_algorithm_from_text("Binary search in a sorted array")
    for cat, data in res['category_recommendations'].items():
        score = data['relevance_score']
        assert 50.0 <= score <= 98.5, f"Score {score} out of bounds for category {cat}"
    print("PASSED ✅")


def test_11_explanation_generation():
    """Test problem-specific explanation generation."""
    print("Testing 11. Problem-specific explanation generation...", end=" ")
    prompt = "I have a sorted array with 1,000,000 numbers and need fast logarithmic lookup."
    prob_rep = analyze_natural_language_problem(prompt)
    algo = ALGORITHM_KNOWLEDGE_BASE[0] # Binary Search
    
    explanation = generate_problem_specific_explanation(prob_rep, algo)
    assert isinstance(explanation, str)
    assert len(explanation) > 20
    assert "Binary Search" in explanation or algo['name'] in explanation
    print("PASSED ✅")


if __name__ == "__main__":
    print("=" * 60)
    print("SuggestAlgo AI — Running Mode 2 Category Test Suite")
    print("=" * 60)
    
    tests = [
        test_1_category_list_validation,
        test_2_knowledge_base_schema,
        test_3_minimum_algorithm_coverage,
        test_4_category_filtering,
        test_5_multi_category_recommendation,
        test_6_empty_category_selection,
        test_7_text_only_mode2,
        test_8_image_ocr_mode2,
        test_9_text_plus_image_mode2,
        test_10_ranking_score_calculation,
        test_11_explanation_generation
    ]
    
    passed = 0
    failed = 0
    
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED ❌ ({str(e)})")
            failed += 1
            
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests.")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)
