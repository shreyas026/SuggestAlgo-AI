import json
import os
import pandas as pd
import numpy as np

# Load knowledge base
kb_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithm_knowledge_base.json')
with open(kb_path, 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

print(f"Loaded {len(knowledge_base)} algorithms from knowledge base.")

USER_QUERY_STYLES = [
    "I need to {verb} {pattern}. Constraints: {time_comp} time and {space_comp} space.",
    "{prefix} how should I approach {pattern} using {ds_str}?",
    "What is the best way to handle {pattern} in a production environment?",
    "We have a problem where we must {verb} {pattern}. Data is {input_char}.",
    "Looking for a {category} technique to solve {pattern} for {use_case}.",
    "Can someone recommend an algorithm to {verb} {use_case}? Requirements: {time_comp}.",
    "Building a module to process {pattern}. Need efficiency of {time_comp} and memory {space_comp}.",
    "{prefix} {pattern} is causing performance degradation. How to optimize?",
    "Is {algo_name} the right choice for {pattern} when using {ds_str}?",
    "Task: {verb} {use_case} with input characteristics ({input_char}).",
    "Which algorithm handles {pattern} best under constraint: {constraint}?",
    "Implementing {pattern} for {use_case}. What data structures should be paired with it?",
    "We are encountering an issue with {pattern} in {category}. Any standard algorithm?",
    "{prefix} we need to partition and process data for {pattern}.",
    "Algorithmic bottleneck: {verb} {pattern}. Goal is {time_comp} bound.",
    "Drafting architecture for {use_case}. Need a {category} algorithm operating on {ds_str}.",
    "Comparison needed: evaluating options to {verb} {pattern} effectively.",
    "Real-time requirement: {verb} {pattern} with minimum memory footprint {space_comp}.",
    "How to implement {pattern} when input data satisfies {input_char}?",
    "Our goal is to solve {pattern} using {algo_name} principles."
]

CONTEXT_PREFIXES = [
    "In our system,",
    "For our web server,",
    "When processing large datasets,",
    "In an embedded environment,",
    "For a cloud microservice,",
    "In a data pipeline,",
    "When optimizing database queries,",
    "In a game engine,",
    "For a mobile application,",
    "In a network router,"
]

PROBLEM_VERBS = [
    "find",
    "locate",
    "compute",
    "identify",
    "select",
    "evaluate",
    "determine",
    "retrieve",
    "organize",
    "calculate"
]

DATASET_RECORDS = []
unique_descriptions = set()

np.random.seed(42)

for algo in knowledge_base:
    algo_id = algo["id"]
    algo_name = algo["name"]
    category = algo["category"]
    patterns = algo.get("problem_patterns", [algo_name.lower()])
    ds_list = algo.get("data_structures", ["Array"])
    ds_str = ", ".join(ds_list)
    time_comp = algo.get("time_complexity", "O(n)")
    space_comp = algo.get("space_complexity", "O(1)")
    use_cases = algo.get("best_use_cases", [f"solving {algo_name} tasks"])
    input_chars = algo.get("input_characteristics", ["standard input"])
    constraints = algo.get("constraints", ["standard constraints"])
    
    algo_count = 0
    attempts = 0
    while algo_count < 50 and attempts < 10000:
        attempts += 1
        prefix = CONTEXT_PREFIXES[attempts % len(CONTEXT_PREFIXES)]
        verb = PROBLEM_VERBS[attempts % len(PROBLEM_VERBS)]
        pattern = patterns[attempts % len(patterns)]
        use_case = use_cases[attempts % len(use_cases)].lower()
        input_char = input_chars[attempts % len(input_chars)].lower()
        constraint = constraints[attempts % len(constraints)].lower()
        style = USER_QUERY_STYLES[attempts % len(USER_QUERY_STYLES)]
        
        prompt = style.format(
            prefix=prefix,
            verb=verb,
            pattern=pattern,
            time_comp=time_comp,
            space_comp=space_comp,
            ds_str=ds_str,
            use_case=use_case,
            category=category.lower(),
            algo_name=algo_name if attempts % 4 == 0 else "the selected approach",
            input_char=input_char,
            constraint=constraint
        )
        
        # Add realistic phrasing noise (synonym variation, prompt reordering)
        if attempts % 3 == 1:
            prompt = f"Problem statement: {prompt}"
        elif attempts % 3 == 2:
            prompt = f"{prompt} Need recommendation."
            
        prompt = " ".join(prompt.split())
        
        if prompt not in unique_descriptions:
            unique_descriptions.add(prompt)
            algo_count += 1
            rec_id = f"{algo_id}_{algo_count:02d}"
            
            record = {
                "problem_id": rec_id,
                "problem_description": prompt,
                "category": category,
                "algorithm": algo_name,
                "data_structures": ds_str,
                "input_characteristics": input_char,
                "constraints": constraint,
                "optimization_goal": f"Time: {time_comp}, Space: {space_comp}"
            }
            DATASET_RECORDS.append(record)

print(f"Generated {len(DATASET_RECORDS)} total dataset records across {len(knowledge_base)} algorithms.")
print(f"Total unique problem descriptions: {len(unique_descriptions)}.")

df_full = pd.DataFrame(DATASET_RECORDS)
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
full_csv_path = os.path.join(data_dir, 'problem_algorithm_dataset.csv')
df_full.to_csv(full_csv_path, index=False)
print(f"Saved full dataset ({len(df_full)} rows) to {full_csv_path}.")
