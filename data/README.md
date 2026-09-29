# SuggestAlgo AI — Independent Algorithm Recommendation Benchmark Dataset

## Dataset Overview
The **SuggestAlgo AI Independent Algorithm Recommendation Benchmark Dataset** is a structured multi-class text classification dataset designed for evaluating supervised machine learning and natural language processing models in algorithm recommendation tasks.

- **Total Samples**: 5,600 problem descriptions
- **Number of Algorithms (Classes)**: 112 unique algorithms
- **Primary Categories**: 10 distinct categories (Searching, Sorting, Dynamic Programming, Graph Algorithms, Greedy Algorithms, Backtracking, Divide and Conquer, Bit Manipulation, String Algorithms, Machine Learning & Mathematical Algorithms)
- **Samples per Class**: Exactly 50 unique problem descriptions per algorithm

---

## Dataset Splits & Physical Files
To prevent data leakage and support rigorous academic experimentation, the dataset is physically partitioned into three separate files using a **70% / 15% / 15%** stratified split (`random_state=42`):

1. **Training Set (`data/train_algorithm_dataset.csv`)**: 3,920 samples (70%)
2. **Validation Set (`data/validation_algorithm_dataset.csv`)**: 840 samples (15%)
3. **Final Untouched Test Set (`data/test_algorithm_dataset.csv`)**: 840 samples (15%)

*Note: Backward-compatible copies (`data/train.csv` and `data/test.csv`) are also maintained for legacy compatibility.*

---

## Schema
Each record contains the following structured attributes:
- `problem_id`: Unique identifier (e.g. `binary_search_01`)
- `problem_description`: Natural language problem statement or user engineering request
- `category`: Primary algorithm category (e.g. `Searching`)
- `algorithm`: Target algorithm class (e.g. `Binary Search`)
- `data_structures`: Associated data structures (e.g. `Array, Sorted List`)
- `input_characteristics`: Input characteristics (e.g. `Sorted numerical sequence`)
- `constraints`: Performance or structural constraints
- `optimization_goal`: Time and space complexity targets

---

## Data Leakage Prevention & Preprocessing Rules
1. **Zero Text Overlap**: Strict string deduplication guarantees zero exact or template overlap across `train`, `validation`, and `test` splits.
2. **Strict Vectorization Isolation**: Feature transformers (`TfidfVectorizer`) are fitted **EXCLUSIVELY** on the training set (`data/train_algorithm_dataset.csv`).
3. **Untouched Final Test Set**: The test set was evaluated exactly **once** after model hyperparameter selection on the validation set.

---

## Transparency Disclaimer
> **Academic Benchmark Disclaimer**: *This dataset is a curated benchmark derived from the SuggestAlgo AI algorithm knowledge base. It is designed as a rigorous proof-of-concept recommendation benchmark for comparing supervised classifiers and does not represent an externally collected real-world software usage log.*
