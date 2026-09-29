# 🧠 SuggestAlgo AI — Explainable AI Assistant for Algorithm Selection

<p align="center">
  <strong>An intelligent, category-aware Explainable AI system that recommends optimal algorithms from CSV datasets, computational problems, and AI project ideas.</strong>
</p>

---

## 🌟 Overview & Operational Modes

**SuggestAlgo AI** is an academic Explainable AI (XAI) algorithm recommendation assistant that operates across two primary input modes:

### 📊 Mode 1 — Dataset / CSV Recommendation & ML Benchmarking
- **Workflow**: `Dataset CSV -> Profiling -> Task Auto-Detection -> Meta-Feature Extraction -> Candidate Model Pool -> 5-Fold Cross Validation -> Metrics -> SHAP Explainability -> CSV Export`
- **Supported Task Types**: Automatically detects **Classification** (binary & multi-class) and **Regression** tasks.
- **Classification Candidate Pool**: Random Forest, Decision Tree, Logistic Regression, SVC, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, SGD Classifier.
- **Regression Candidate Pool**: Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree Regressor, Random Forest Regressor, Extra Trees Regressor, Gradient Boosting Regressor, Support Vector Regressor (SVR), XGBoost Regressor.
- **Validation & Metrics**: 5-Fold Cross Validation benchmark (Accuracy, F1-Score, Precision, Recall, R² Score, MAE, RMSE).
- **Explainability**: SHAP (SHapley Additive exPlanations) summary plots and feature importance rankings.

### 💡 Mode 2 — Category-Aware Problem & Project Idea Recommendation
- **Workflow**: `Text/Image Input -> OCR -> Structured Problem Representation -> Category Selection -> Semantic Retrieval / Supervised ML Classifier -> Category-Specific Recommendations`
- **Supervised ML Model Pipeline**: `Curated Dataset (1,792 samples) -> 80/20 Stratified Split -> TF-IDF (1-2 N-Grams) -> Logistic Regression -> Unseen Test Set Evaluation (77.16% Accuracy, 75.94% Macro F1)`
- **Standardized Categories**: 10 primary categories (Searching, Sorting, Classification, Regression, Graph Algorithms, Tree Algorithms, Dynamic Programming, Greedy Algorithms, Backtracking, Machine Learning / AI).
- **Knowledge Base**: 112+ algorithms with full structured metadata (10-15 algorithms per primary category).
- **OCR Integration**: Supports image upload with graceful fallback if system Tesseract binary is uninstalled.
- **Scoring Definition**: Uses a transparent **Relevance / Suitability Score** combining semantic cosine embeddings and requirement matching alongside **Supervised ML Model Probabilities**.

> *The current supervised Mode 2 model is trained on a curated benchmark generated from the project's algorithm knowledge base. It is intended as a proof-of-concept recommendation model and does not claim to represent real-world algorithm usage.*

---


## 🔑 Terminology & Concepts

- **Algorithm**: A step-by-step computational procedure or data structure technique for solving a problem (e.g., *Binary Search*, *Dijkstra's Algorithm*, *QuickSort*).
- **Machine Learning Algorithm**: A data-driven model learning parameters from features (e.g., *Random Forest*, *Logistic Regression*, *Gradient Boosting*).
- **Trained ML Model**: An instantiated algorithm fitted on a specific dataset with optimized weights and hyperparameters.
- **Relevance / Suitability Score**: A normalized ranking metric (50%–98.5%) measuring semantic query similarity and problem requirement compatibility. *It is a relative suitability ranking metric, NOT a statistical probability of correctness.*

---

## 🏗️ System Architecture

```
                                 SUGGESTALGO AI
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
     📊 MODE 1: DATASET / CSV                              💡 MODE 2: PROBLEM / IDEA
            │                                                     │
     CSV File + Target                                    Text / Image Problem Statement
            │                                                     │
     Task Auto-Detection                                  OCR Extraction (Pytesseract)
 (Classification / Regression)                                    │
            │                                           Structured Problem Representation
    Dataset Profiling                                   (Data Struct, Constraints, Size, Goal)
            │                                                     │
   Meta-Feature Extraction                                Category Selection Filter
   (Skewness, Entropy, Ratio)                               (10 Primary Categories)
            │                                                     │
   KNN Meta-Learning Match                             Semantic Sentence Transformers / TF-IDF
            │                                                     │
 5-Fold Cross-Validation                               Transparent Suitability Scoring
  (9 Classifiers / 11 Regressors)                         (Semantic Match + Requirement Boost)
            │                                                     │
   SHAP Explainability Plots                             Category-Specific Top Recommendations
            │                                            (Why Selected, Complexity, Python Code)
   CSV Download Benchmark
```

---

## 🏷️ Standardized 10 Primary Categories

SuggestAlgo AI maps all algorithms into 10 primary categories:

1. **Searching**: Linear Search, Binary Search, Jump Search, Interpolation Search, Exponential Search, Fibonacci Search, Ternary Search, Hash Table Lookup, BFS Search, DFS Search, Two Pointer Search, Sliding Window Search.
2. **Sorting**: Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quick Sort, Heap Sort, Counting Sort, Radix Sort, Bucket Sort, Shell Sort, TimSort.
3. **Classification**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Extra Trees Classifier, Gradient Boosting Classifier, AdaBoost Classifier, SVC, KNN, Naive Bayes Classifier, SGD Classifier, XGBoost Classifier.
4. **Regression**: Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree Regressor, Random Forest Regressor, Extra Trees Regressor, Gradient Boosting Regressor, SVR, XGBoost Regressor.
5. **Graph Algorithms**: BFS Graph, DFS Graph, Dijkstra's Algorithm, Bellman-Ford, Floyd-Warshall, A* Search, Prim's Algorithm, Kruskal's Algorithm, Topological Sort, Union-Find, Johnson's Algorithm.
6. **Tree Algorithms**: Binary Search Tree, AVL Tree, Red-Black Tree, Binary Tree Traversal, Level Order Traversal, Trie (Prefix Tree), Binary Heap, Segment Tree, Fenwick Tree (BIT), Lowest Common Ancestor.
7. **Dynamic Programming**: 0/1 Knapsack, Unbounded Knapsack, Coin Change, Longest Common Subsequence (LCS), Longest Increasing Subsequence (LIS), Edit Distance, Matrix Chain Multiplication, Rod Cutting, Grid Path DP, Interval DP, Bitmask DP.
8. **Greedy Algorithms**: Activity Selection, Fractional Knapsack, Huffman Coding, Job Sequencing with Deadlines, Interval Scheduling, Jump Game, Gas Station Problem, Kruskal's MST (Greedy), Prim's MST (Greedy), Dijkstra (Greedy), Minimum Spanning Tree Strategy.
9. **Backtracking**: N-Queens, Sudoku Solver, Rat in a Maze, Permutations, Combinations, Subsets (Power Set), Combination Sum, Graph Coloring, Hamiltonian Path, Word Search Grid, Knight's Tour.
10. **Machine Learning / AI**: K-Means, DBSCAN, Hierarchical Clustering, PCA, Naive Bayes ML, Multilayer Perceptron (MLP), CNN, RNN, LSTM, Transformer, Content-Based Recommendation, Collaborative Filtering, Hybrid Recommendation.

---

## ⚙️ Installation & Execution

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/shreyas026/SuggestAlgo-AI.git
cd SuggestAlgo-AI
pip install -r requirements.txt
```

### 2. Optional: Tesseract OCR Setup (for Image-to-Text in Mode 2)

- **Windows**: Download Tesseract installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add `C:\Program Files\Tesseract-OCR` to system PATH.
- **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

*Note: If Tesseract is not installed, Mode 2 remains fully operational via text input, and gracefully displays diagnostic guidance.*

### 3. Run Test Suite

```bash
python -X utf8 tests/test_pipeline.py
python -X utf8 tests/test_mode2_category.py
```

### 4. Launch Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`.

---

## 🔬 Academic Foundation & Attribution

SuggestAlgo AI adapts its foundational meta-learning algorithm candidate pool and explainability framework from:
- **AMLBID**: Automated Machine Learning with Big Data Meta-Learning & Explainability by *LeMGarouani et al.* ([GitHub Repo](https://github.com/LeMGarouani/AMLBID)).

---

## 🌐 Public Deployment & Repository

- **GitHub Repository**: [https://github.com/shreyas026/SuggestAlgo-AI](https://github.com/shreyas026/SuggestAlgo-AI)
- **Live Application**: [https://suggestalgo-ai.streamlit.app](https://suggestalgo-ai.streamlit.app)
