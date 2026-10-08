# 🧠 SuggestAlgo AI — Explainable AI Assistant for Algorithm Selection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/shreyas026/SuggestAlgo-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/shreyas026/SuggestAlgo-AI/actions)
[![Stars](https://img.shields.io/github/stars/shreyas026/SuggestAlgo-AI?style=social)](https://github.com/shreyas026/SuggestAlgo-AI/stargazers)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)

**An intelligent, category-aware Explainable AI (XAI) system that recommends optimal algorithms from CSV datasets, computational problems, and AI project ideas.**

---

## 🌟 Overview & Operational Modes

### 📊 Mode 1 — Dataset / CSV Recommendation & ML Benchmarking
- **Workflow**: Dataset CSV → Profiling → Task Auto-Detection → Meta-Feature Extraction → Candidate Model Pool → 5-Fold CV → Metrics → SHAP Explainability → CSV Export
- **Task Types**: Classification (binary & multi-class) & Regression
- **Candidate Pool**: 9 classifiers + 11 regressors (RF, XGBoost, SVM, etc.)
- **Validation**: 5-Fold Cross Validation (Accuracy, F1, Precision, Recall, R², MAE, RMSE)
- **Explainability**: SHAP summary plots & feature importance rankings

### 💡 Mode 2 — Category-Aware Problem & Project Idea Recommendation
- **Workflow**: Text/Image Input → OCR → Structured Problem → Category Selection → Semantic/ML Retrieval → Category-Specific Recommendations
- **Supervised ML Pipeline**: 1,792 samples → TF-IDF → Logistic Regression → 77.16% Accuracy, 75.94% Macro F1
- **Categories**: 10 primary (Searching, Sorting, Classification, Regression, Graph, Tree, DP, Greedy, Backtracking, ML/AI)
- **Knowledge Base**: 112+ algorithms with full metadata
- **OCR**: Tesseract integration with graceful fallback

---

## 🏗️ System Architecture

`
                              SUGGESTALGO AI
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
     📊 MODE 1: DATASET / CSV                          💡 MODE 2: PROBLEM / IDEA
            │                                               │
     CSV File + Target                              Text / Image Input
            │                                               │
     Task Auto-Detection                            OCR Extraction
     (Classification / Regression)                       │
            │                                    Structured Problem
     Dataset Profiling                                Representation
            │                                               │
     Meta-Feature Extraction                       Category Selection
     (Skewness, Entropy, Ratio)                      (10 Categories)
            │                                               │
     KNN Meta-Learning Match                     Semantic / TF-IDF
            │                                               │
     5-Fold CV (9 Clf / 11 Reg)                   Transparent Scoring
            │                                               │
     SHAP Explainability                         Category-Specific
     CSV Download                                 Recommendations
`

---

## 🏷️ Standardized 10 Primary Categories

1. **Searching**: Linear, Binary, Jump, Interpolation, Exponential, Fibonacci, Ternary, Hash Table, BFS, DFS, Two Pointer, Sliding Window
2. **Sorting**: Bubble, Selection, Insertion, Merge, Quick, Heap, Counting, Radix, Bucket, Shell, TimSort
3. **Classification**: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, SVC, KNN, Naive Bayes, SGD, XGBoost
4. **Regression**: Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVR, XGBoost
5. **Graph Algorithms**: BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall, A*, Prim, Kruskal, Topological Sort, Union-Find, Johnson
6. **Tree Algorithms**: BST, AVL, Red-Black, Traversals, Trie, Heap, Segment Tree, Fenwick Tree, LCA
7. **Dynamic Programming**: 0/1 Knapsack, Coin Change, LCS, LIS, Edit Distance, Matrix Chain, Rod Cutting, Grid Path, Interval DP, Bitmask DP
8. **Greedy Algorithms**: Activity Selection, Fractional Knapsack, Huffman Coding, Job Sequencing, Interval Scheduling, Jump Game, Gas Station, MST Strategies
9. **Backtracking**: N-Queens, Sudoku, Rat in Maze, Permutations, Combinations, Subsets, Combination Sum, Graph Coloring, Hamiltonian Path, Word Search, Knight's Tour
10. **Machine Learning / AI**: K-Means, DBSCAN, Hierarchical Clustering, PCA, Naive Bayes, MLP, CNN, RNN, LSTM, Transformer, Recommendation Systems

---

## ⚙️ Installation & Execution

### 1. Clone & Install Dependencies
`ash
git clone https://github.com/shreyas026/SuggestAlgo-AI.git
cd SuggestAlgo-AI
pip install -r requirements.txt
`

### 2. Optional: Tesseract OCR Setup (for Image-to-Text in Mode 2)
- **Windows**: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) → Add to PATH
- **Linux**: sudo apt-get install tesseract-ocr
- **macOS**: rew install tesseract

### 3. Run Test Suite
`ash
python -X utf8 tests/test_pipeline.py
python -X utf8 tests/test_mode2_category.py
`

### 4. Launch Application
`ash
streamlit run app.py
`
Navigate to http://localhost:8501

---

## 🔬 Academic Foundation
Adapted from **AMLBID**: Automated Machine Learning with Big Data Meta-Learning & Explainability by *LeMGarouani et al.* ([GitHub](https://github.com/LeMGarouani/AMLBID))

---

## 🌐 Links
- **Repository**: https://github.com/shreyas026/SuggestAlgo-AI
- **Live Demo**: https://suggestalgo-ai.streamlit.app

---

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License
MIT License — see [LICENSE](LICENSE)
