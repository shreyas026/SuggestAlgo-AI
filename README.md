# 🧠 SuggestAlgo AI — Explainable AI Assistant for Algorithm Selection

<p align="center">
  <strong>An intelligent, explainable system that recommends optimal algorithms from CSV datasets, computational problems, and AI project ideas.</strong>
</p>

---

## 🌟 Overview & Key Features

**SuggestAlgo AI** is an academic Explainable AI (XAI) recommendation assistant that solves the algorithm selection problem across two distinct operational modes:

1. **📊 Mode 1 — Dataset / CSV Recommendation & ML Benchmarking**:
   - Accepts user CSV files or pre-loaded datasets (Iris, Breast Cancer, Wine).
   - Profiles statistical dataset meta-features (instances, features, entropy, imbalance ratio, correlation).
   - Applies meta-learning kNN similarity matching against historical knowledge to recommend the optimal ML algorithm.
   - Evaluates **9 candidate classifiers** (*Random Forest, Decision Tree, Logistic Regression, SVC, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, SGD Classifier*).
   - Runs **5-Fold Stratified Cross Validation** alongside test split metrics.
   - Computes **SHAP (SHapley Additive exPlanations)** global feature importances.
   - Generates an academic **ML Proof** breakdown and downloadable CSV report.

2. **💡 Mode 2 — Problem / Question / Project Idea Recommendation**:
   - Accepts natural-language descriptions of coding problems (e.g., searching a sorted array, weighted graph routing) or AI project ideas (e.g., customer churn, email spam classification, movie recommendation system).
   - Leverages **Sentence Transformers** (`all-MiniLM-L6-v2`) with TF-IDF fallback to compute semantic similarity against a rich algorithm knowledge base covering 25+ categories.
   - Extracts structured problem properties: Problem Category, Target Data Structure, Time Complexity $O(\cdot)$, Space Complexity, Requirements, and Limitations.
   - Generates ranked match confidence percentages, trade-off comparisons, targeted follow-up questions for sparse prompts, and Python code implementation templates.

---

## 🏗️ System Architecture

```
                    SUGGESTALGO AI
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       📊 DATASET MODE          💡 PROBLEM MODE
              │                       │
              ▼                       ▼
       CSV + Target             Natural Language
              │                       │
              ▼                       ▼
       Data Profiling          Text Processing
              │                       │
              ▼                       ▼
       Meta Features          Sentence Embedding
              │                       │
              ▼                       ▼
       KNN Similarity          Semantic Similarity
              │                       │
              ▼                       ▼
       ML Recommendation       Algorithm Ranking
              │                       │
              ▼                       ▼
      9 Model Benchmark          Recommendation
              │                       │
              ▼                       ▼
       Best ML Model          Explanation + Complexity
              │
              ▼
          SHAP / XAI
```

---

## ⚙️ Installation & Local Execution

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/shreyas026/SuggestAlgo-AI.git
cd SuggestAlgo-AI
pip install -r requirements.txt
```

### 2. Run Test Suite (10/10 Tests)

```bash
python tests/test_pipeline.py
```

### 3. Launch Streamlit Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🔬 Academic Attribution & Research Foundation

SuggestAlgo AI adapts its foundational meta-learning algorithm candidate pool and explainability framework from:
- **AMLBID**: Automated Machine Learning with Big Data Meta-Learning & Explainability by *LeMGarouani et al.* ([GitHub Repo](https://github.com/LeMGarouani/AMLBID)).

---

## 🌐 Public Deployment & Access

- **GitHub Repository**: [https://github.com/shreyas026/SuggestAlgo-AI](https://github.com/shreyas026/SuggestAlgo-AI)
- **Live Streamlit App**: [https://suggestalgo-ai.streamlit.app](https://suggestalgo-ai.streamlit.app)
