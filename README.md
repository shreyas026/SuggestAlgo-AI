# 🧠 SuggestAlgo AI — Explainable AI for Machine Learning Algorithm Selection

<p align="center">
  <strong>An intelligent, explainable system that recommends the most suitable machine learning algorithm for your dataset using meta-learning and provides transparent AI explanations.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-methodology">Methodology</a> •
  <a href="#-features">Features</a> •
  <a href="#-deployment">Deployment</a>
</p>

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Motivation](#-motivation)
3. [Objectives](#-objectives)
4. [Existing System / Research Foundation](#-existing-system--research-foundation)
5. [Proposed System — SuggestAlgo AI](#-proposed-system--suggestalgo-ai)
6. [Architecture](#-architecture)
7. [Machine Learning Methodology](#-machine-learning-methodology)
8. [Dataset & Meta-Features](#-dataset--meta-features)
9. [Candidate Algorithms](#-candidate-algorithms)
10. [Algorithm Selection Approach](#-algorithm-selection-approach)
11. [Explainable AI (XAI) Methodology](#-explainable-ai-xai-methodology)
12. [SHAP Integration](#-shap-integration)
13. [Model Evaluation](#-model-evaluation)
14. [Results](#-results)
15. [Screenshots](#-screenshots)
16. [Installation](#-installation)
17. [Running Locally](#-running-locally)
18. [Deployment](#-deployment)
19. [Repository Structure](#-repository-structure)
20. [Source Attribution](#-source-attribution)
21. [Limitations](#-limitations)
22. [Future Enhancements](#-future-enhancements)
23. [References](#-references)

---

## 🎯 Problem Statement

Selecting the right machine learning algorithm for a given dataset is a challenging task that typically requires significant ML expertise. Practitioners must consider dataset characteristics (size, dimensionality, class balance, feature types) and match them to algorithm strengths — a process that is time-consuming and error-prone.

**SuggestAlgo AI** addresses this challenge by automating algorithm selection using meta-learning and providing explainable AI (XAI) explanations for every recommendation.

---

## 💡 Motivation

- Manual algorithm selection is time-consuming and requires expert knowledge
- No-Free-Lunch theorem: no single algorithm is universally best
- Existing AutoML tools often lack transparency in their recommendations
- Academic and industry need for interpretable ML pipelines
- The AMLBID framework demonstrates that meta-learning can effectively automate algorithm selection with explainability

---

## 🎯 Objectives

1. Build a web-based application for automated ML algorithm selection
2. Implement meta-learning using dataset meta-feature extraction and KNN-based similarity matching
3. Evaluate multiple candidate ML algorithms on user-provided datasets
4. Provide Explainable AI explanations using SHAP and feature importance analysis
5. Distinguish clearly between algorithm selection explanations and model prediction explanations
6. Deploy as a publicly accessible academic demonstration

---

## 📚 Existing System / Research Foundation

### AMLBID (Automating Machine-Learning model selection and configuration with Big Industrial Data)

**SuggestAlgo AI** is built upon concepts and methodology from the **AMLBID** framework:

- **Repository**: [https://github.com/LeMGarouani/AMLBID](https://github.com/LeMGarouani/AMLBID)
- **Authors**: LeMGarouani et al.
- **Description**: A meta-learning based framework for automating algorithm selection and hyperparameter tuning in supervised machine learning, with built-in explainability.

#### AMLBID Key Concepts Used:
| AMLBID Component | SuggestAlgo AI Adaptation |
|---|---|
| `MetafeaturesExtractor.py` — Extracts dataset meta-features (nr_classes, nr_instances, skewness, kurtosis, correlations) | Reimplemented with extended meta-features |
| `loader.py` — KNN-based nearest neighbor search in knowledge base | Adapted with KNN meta-learning and weighted voting |
| `KnowledgeBase/` — CSV-based knowledge base of datasets and best algorithms | Built compact knowledge base encoding dataset→algorithm mappings |
| `Explainer/` — SHAP-based explainability with Dash dashboard | Reimplemented using SHAP with Streamlit |
| Candidate algorithms (RandomForest, SVC, XGBoost, etc.) | Same algorithm pool from `AMLBID/loader.py` |

---

## 🏗 Proposed System — SuggestAlgo AI

SuggestAlgo AI is a **customized, functional academic project** that:

1. **Does NOT simply clone** AMLBID — it reimplements core concepts in a Streamlit-based architecture
2. **Adds** interactive dataset upload, profiling, and visualization
3. **Extends** the meta-feature set with additional statistical and class-balance features
4. **Replaces** the Dash-based dashboard with a modern Streamlit interface
5. **Maintains** the core meta-learning methodology for scientific integrity

---

## 🏛 Architecture

```
User
  ↓
Streamlit Web App (app.py)
  ↓
CSV Upload / Sample Dataset
  ↓
Dataset Validation (data_loader.py)
  ↓
Dataset Profiling (profiling.py)
  ↓
Preprocessing (preprocessing.py)
  ↓
Meta-Feature Extraction (algorithm_selection.py)
  ↓
KNN Meta-Learning Algorithm Selection
  ↓
Candidate Model Training & Evaluation (model_evaluation.py)
  ↓
Explainability — SHAP & Feature Importance (explainability.py)
  ↓
Interactive Dashboard with Results
```

### Key Distinction:

| Component | Purpose |
|-----------|---------|
| **Candidate ML Algorithms** | The actual models (Random Forest, SVM, XGBoost, etc.) trained on user data |
| **Meta-Learning Selection Mechanism** | The KNN-based system that recommends which algorithm to use based on dataset characteristics |

**AMLBID performs algorithm selection among supported candidate ML models** — it is not itself a classifier.

---

## 🔬 Machine Learning Methodology

### Step-by-Step Process:

| Step | Description |
|------|-------------|
| 1. Dataset Acquisition | User uploads CSV or selects a sample dataset |
| 2. Dataset Validation | Check for valid format, sufficient rows/columns, valid target |
| 3. Dataset Profiling | Compute statistics: dimensions, types, missing values, class distribution |
| 4. Preprocessing | Imputation (mean for numerical, mode for categorical), encoding, scaling |
| 5. Meta-Feature Extraction | Compute 20+ meta-features characterizing the dataset |
| 6. Algorithm Selection | KNN-based meta-learning finds similar datasets, recommends best algorithm |
| 7. Candidate Training | Train all 9 candidate algorithms with stratified train/test split |
| 8. Performance Evaluation | Compute accuracy, precision, recall, F1 score for each |
| 9. Explainability | SHAP values and feature importance for the best model |
| 10. Deployment | Serve via Streamlit with interactive visualizations |

---

## 📊 Dataset & Meta-Features

### Meta-Features Extracted (inspired by AMLBID's MetafeaturesExtractor.py):

| Meta-Feature | Description | Source |
|---|---|---|
| `nr_classes` | Number of target classes | AMLBID |
| `nr_instances` | Number of data points | AMLBID |
| `log_nr_instances` | Log-transformed instance count | AMLBID |
| `nr_features` | Number of input features | AMLBID |
| `log_nr_features` | Log-transformed feature count | AMLBID |
| `dataset_ratio` | Features / Instances ratio | AMLBID |
| `skewness_mean/std/min/max` | Distribution skewness statistics | AMLBID |
| `kurtosis_mean/std/min/max` | Distribution kurtosis statistics | AMLBID |
| `corr_mean/std/min/max` | Pairwise correlation statistics | AMLBID |
| `variance_mean/std` | Feature variance statistics | Added |
| `class_entropy` | Entropy of class distribution | Added |
| `class_imbalance` | Max/min class ratio | Added |
| `instances_to_features` | Instance-to-feature ratio | Added |

### Sample Datasets Included:

| Dataset | Instances | Features | Classes | Source |
|---------|-----------|----------|---------|--------|
| Iris | 150 | 4 | 3 | sklearn |
| Breast Cancer | 569 | 30 | 2 | sklearn |
| Wine | 178 | 13 | 3 | sklearn |

---

## 🤖 Candidate Algorithms

These are the same algorithms used in AMLBID (from `AMLBID/loader.py` imports):

| Algorithm | Type | Key Strength |
|-----------|------|--------------|
| Random Forest | Ensemble (Bagging) | Robust, handles high dimensions |
| Decision Tree | Tree | Interpretable, fast |
| Logistic Regression | Linear | Fast, good for linearly separable data |
| SVC | Kernel-based | Effective in high-dimensional spaces |
| Extra Trees | Ensemble (Bagging) | Reduced variance, fast |
| Gradient Boosting | Ensemble (Boosting) | High accuracy, sequential correction |
| AdaBoost | Ensemble (Boosting) | Focus on hard examples |
| XGBoost | Ensemble (Boosting) | State-of-the-art tabular performance |
| SGD Classifier | Linear | Scalable to large datasets |

---

## 🎯 Algorithm Selection Approach

### Meta-Learning with KNN (adapted from AMLBID):

1. **Extract meta-features** from the user's dataset (20+ statistical, structural, and distributional features)
2. **Normalize** meta-features using StandardScaler for fair distance comparison
3. **Find K nearest neighbors** (K=3) in the knowledge base using Euclidean distance
4. **Weighted voting**: Algorithms from nearest neighbors vote, weighted by inverse distance
5. **Recommend** the algorithm with the highest weighted vote

This is **NOT** hard-coded selection. The recommendation changes based on actual dataset characteristics.

---

## 🔍 Explainable AI (XAI) Methodology

### Two Distinct Explanation Types:

#### 1. Algorithm Selection Explanation
- Why the meta-learning system recommended a particular algorithm
- Based on dataset meta-features and similarity to known dataset profiles
- Shows which dataset characteristics influenced the recommendation

#### 2. Model Prediction Explanation
- Why the trained model makes specific predictions
- Uses SHAP values and feature importance
- Shows which features are most influential for the model's decisions

> **Important**: We clearly distinguish these two types of explanation. SHAP explains the model's predictions, not the algorithm selection process, unless specifically implemented.

---

## 📊 SHAP Integration

| Model Type | SHAP Explainer | Speed |
|-----------|---------------|-------|
| Tree-based (RF, DT, ET, GB, XGB) | TreeExplainer | Fast |
| Linear (LR, SGD) | LinearExplainer | Fast |
| Others (SVC) | KernelExplainer | Slow (sampled) |

- Maximum SHAP samples: 500 (for performance)
- Graceful fallback to `feature_importances_` if SHAP fails
- Both summary bar plots and beeswarm plots generated

---

## 📈 Model Evaluation

- **Split**: Stratified train/test split (70/30 default)
- **Random State**: 42 (for reproducibility)
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Averaging**: Weighted/macro for multiclass, binary for binary classification
- **Visualization**: Confusion matrix, radar charts, bar charts

---

## 📊 Results

Results are **computed at runtime** from actual model training — never hard-coded.

Example results on Iris dataset (actual values will vary):

| Algorithm | Accuracy | F1 Score | Training Time |
|-----------|----------|----------|---------------|
| *Varies per run* | *Actual computed* | *Actual computed* | *Actual measured* |

> "The best-performing candidate on this dataset under the selected evaluation metric" — NOT "the universally best algorithm."

---

## 📸 Screenshots

*Screenshots should be captured from the running application demonstrating:*
1. Dataset upload interface
2. Dataset profiling dashboard
3. Algorithm recommendation card
4. Model comparison table and charts
5. SHAP feature importance plots
6. Results summary

---

## ⚡ Installation

### Prerequisites
- Python 3.9+ 
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/shreyas026/SuggestAlgo-AI.git
cd SuggestAlgo-AI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Locally

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Quick Demo:
1. Select "Iris (Classification - 3 classes)" from sample datasets
2. Target column auto-selects to "species"
3. Click "Run Algorithm Selection & Model Evaluation"
4. Explore all sections: profiling, recommendation, comparison, XAI

---

## 🌐 Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set main file: `app.py`
5. Deploy

### Alternative: Render

1. Add `Procfile`: `web: streamlit run app.py --server.port=$PORT --server.headless=true`
2. Deploy via Render dashboard

---

## 📁 Repository Structure

```
SuggestAlgo-AI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── src/
│   ├── __init__.py                 # Package init
│   ├── data_loader.py              # Dataset loading & validation
│   ├── preprocessing.py            # Data preprocessing pipeline
│   ├── profiling.py                # Dataset profiling & statistics
│   ├── algorithm_selection.py      # Meta-learning algorithm selection
│   ├── model_evaluation.py         # ML model training & evaluation
│   ├── explainability.py           # SHAP & XAI explanations
│   └── utils.py                    # Utility functions
├── tests/
│   └── test_pipeline.py            # Pipeline tests
├── sample_data/
│   ├── iris.csv                    # Iris dataset
│   ├── breast_cancer.csv           # Breast cancer dataset
│   └── wine.csv                    # Wine dataset
├── models/                         # Saved model artifacts
├── outputs/                        # Generated outputs
└── notebooks/                      # Jupyter notebooks
```

---

## 🙏 Source Attribution

### Foundation / Reference Implementation

**AMLBID** — Automating Machine-Learning model selection and configuration with Big Industrial Data

- **Authors**: LeMGarouani et al.
- **GitHub**: [https://github.com/LeMGarouani/AMLBID](https://github.com/LeMGarouani/AMLBID)
- **License**: MIT License

### What was reused from AMLBID:
- Meta-feature extraction methodology (MetafeaturesExtractor.py concepts)
- KNN-based meta-learning approach for algorithm selection
- Candidate algorithm pool (loader.py imports)
- SHAP-based explainability approach
- Knowledge base concept for algorithm recommendation

### What was modified/added in SuggestAlgo AI:
- Complete reimplementation in Streamlit (AMLBID uses Dash)
- Extended meta-feature set with class entropy, imbalance metrics
- Interactive dataset upload and profiling interface
- Plotly-based visualizations and comparison charts
- Comprehensive error handling and validation
- Downloadable results
- Streamlit Cloud deployment configuration
- Clean modular architecture (`src/` package structure)
- Automated test suite

---

## ⚠️ Limitations

1. **Classification only**: Currently supports classification tasks only (not regression)
2. **Compact knowledge base**: The meta-learning knowledge base is smaller than AMLBID's full KB (~25MB CSV)
3. **No hyperparameter tuning**: Uses default/reasonable hyperparameters (AMLBID supports full HP tuning)
4. **File format**: Only CSV files supported
5. **Dataset size**: Very large datasets (>100K rows) may cause slow SHAP computation
6. **SHAP for SVC**: KernelExplainer for SVC is computationally expensive
7. **Meta-learning accuracy**: The compact knowledge base may not cover all dataset profiles
8. **No cross-validation**: Uses single train/test split (AMLBID supports cross-validation)

---

## 🚀 Future Enhancements

1. **Regression support**: Extend to regression tasks
2. **Expanded knowledge base**: Integrate AMLBID's full 25MB knowledge base
3. **Hyperparameter tuning**: Add automated hyperparameter optimization
4. **Cross-validation**: Implement k-fold cross-validation
5. **More algorithms**: Add neural networks, LightGBM, CatBoost
6. **Advanced XAI**: LIME explanations, partial dependence plots
7. **PDF reports**: Generate downloadable analysis reports
8. **What-if analysis**: Interactive feature perturbation analysis
9. **Multi-format upload**: Support Excel, JSON, Parquet files
10. **Model persistence**: Save and reload trained models

---

## 📚 References

1. LeMGarouani et al., "AMLBID: Transparent and Auto-explainable AutoML," GitHub, [https://github.com/LeMGarouani/AMLBID](https://github.com/LeMGarouani/AMLBID)
2. Lundberg, S.M. and Lee, S.I., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017
3. Brazdil, P. et al., "Metalearning: Applications to Data Mining," Springer, 2009
4. Wolpert, D.H., "No Free Lunch Theorems for Optimization," IEEE Trans. on Evolutionary Computation, 1997
5. Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," JMLR 12, 2011
6. Chen, T. and Guestrin, C., "XGBoost: A Scalable Tree Boosting System," KDD 2016

---

<p align="center">
  <strong>SuggestAlgo AI</strong> — Built as an academic ML project<br>
  Foundation: AMLBID by LeMGarouani et al.
</p>
