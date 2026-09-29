import os
import sys
import pandas as pd
import numpy as np
import random

sys.path.insert(0, '.')
from src.algorithm_knowledge_base import ALGORITHM_KNOWLEDGE_BASE

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

random.seed(42)
np.random.seed(42)

# Templates and variation parameters per algorithm category/pattern
dataset_rows = []
problem_counter = 1

# Domain contexts for variation
DOMAINS = [
    "in an e-commerce platform database",
    "in a high-frequency trading system",
    "for a mobile navigation app",
    "in a bioinformatics DNA pipeline",
    "for a social network friend graph",
    "in a cloud server log analysis system",
    "for a game engine physics solver",
    "in a digital signal processing pipeline",
    "for a database query optimizer",
    "in a cyber security intrusion detection network",
    "for an autonomous vehicle routing module",
    "in a retail inventory management system"
]

SCALES = [
    "with 1,000 elements",
    "with 10,000 entries",
    "with 100,000 items",
    "with 1,000,000 records",
    "on a massive 10^6 dataset",
    "for small real-time input batches",
    "with 50,000 historical records"
]

# Generate 16 distinct problem descriptions for every single algorithm in ALGORITHM_KNOWLEDGE_BASE
for algo in ALGORITHM_KNOWLEDGE_BASE:
    algo_id = algo["id"]
    algo_name = algo["name"]
    category = algo["category"]
    ds_list = algo.get("data_structures", ["Array"])
    ds_str = ", ".join(ds_list)
    input_chars = ", ".join(algo.get("input_characteristics", ["General Input"]))
    constraints_str = ", ".join(algo.get("constraints", ["Standard Constraints"]))
    best_use = algo.get("best_use_cases", ["General use"])[0]
    desc = algo["description"]
    patterns = algo.get("problem_patterns", [algo_name.lower()])
    
    # Generate 16 distinct variations
    variations = []
    
    # 1-6: Pattern-based problem statements with domain and scale variation
    for idx, pat in enumerate(patterns):
        dom = DOMAINS[idx % len(DOMAINS)]
        sc = SCALES[idx % len(SCALES)]
        v = f"I need to solve a problem involving {pat} {dom} {sc}. What algorithm is best?"
        variations.append((v, best_use))
        
    # Additional variations up to 16
    fillers = [
        (f"How can I implement {algo_name.lower()} to handle {desc.lower()} {DOMAINS[0]}?", "Performance optimization"),
        (f"Need an efficient algorithm that satisfies {constraints_str} operating on a {ds_str}.", best_use),
        (f"Looking for an algorithm that handles {input_chars} {DOMAINS[1]} {SCALES[0]}.", best_use),
        (f"What is the best way to execute {patterns[0]} {DOMAINS[2]} with maximum efficiency?", best_use),
        (f"Need a strategy for {desc.lower()} under strict time complexity bounds.", best_use),
        (f"I am building a module {DOMAINS[3]} requiring {patterns[-1]} {SCALES[2]}.", best_use),
        (f"Require a reliable approach for {algo_name.lower()} using {ds_str} data structures.", best_use),
        (f"How to optimize {patterns[0]} when data characteristics are {input_chars}?", best_use),
        (f"Find the ideal algorithm to process {desc.lower()} {DOMAINS[4]} {SCALES[3]}.", best_use),
        (f"System requires {algo_name.lower()} for {best_use.lower()} {DOMAINS[5]}.", best_use),
        (f"Designing a solution for {patterns[min(1, len(patterns)-1)]} with {constraints_str} {SCALES[1]}.", best_use),
        (f"Which algorithm effectively performs {desc.lower()} on a {ds_str} {DOMAINS[6]}?", best_use),
        (f"Need to implement a robust method for {patterns[0]} {DOMAINS[7]} {SCALES[4]}.", best_use),
        (f"Looking for optimal solution for {algo_name.lower()} addressing {input_chars} {DOMAINS[8]}.", best_use),
        (f"Select the top algorithm for {desc.lower()} in a production environment {SCALES[5]}.", best_use),
        (f"How to execute {patterns[min(2, len(patterns)-1)]} efficiently on {ds_str} {DOMAINS[9]}?", best_use),
    ]
    
    for v_text, opt_goal in fillers:
        if len(variations) >= 16:
            break
        # Avoid duplicate text
        if not any(v_text == existing[0] for existing in variations):
            variations.append((v_text, opt_goal))
            
    # Guarantee exactly 16 variations per algorithm
    for v_text, opt_goal in variations[:16]:
        dataset_rows.append({
            "problem_id": f"PROB_{problem_counter:04d}",
            "problem_description": v_text,
            "category": category,
            "algorithm": algo_name,
            "data_structures": ds_str,
            "input_characteristics": input_chars,
            "constraints": constraints_str,
            "optimization_goal": opt_goal
        })
        problem_counter += 1

df_dataset = pd.DataFrame(dataset_rows)
csv_path = "data/problem_algorithm_dataset.csv"
df_dataset.to_csv(csv_path, index=False)

print(f"Successfully generated dataset: {csv_path}")
print(f"Total Rows: {len(df_dataset)}")
print(f"Total Unique Algorithms: {df_dataset['algorithm'].nunique()}")
print(f"Total Unique Categories: {df_dataset['category'].nunique()}")
print("\nRows per Category:")
print(df_dataset['category'].value_counts())
print("\nRows per Algorithm (First 5):")
print(df_dataset['algorithm'].value_counts().head(5))
