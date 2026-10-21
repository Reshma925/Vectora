# Vectora — Interactive Machine Learning Platform

**Course:** 23CSE301 – Machine Learning  

**Live Application:** [Launch Vectora](https://vectora-m8jqutzw8xnv6njwgymrzu.streamlit.app/)

## About the Project

Vectora is an interactive machine learning application developed as part of the **23CSE301 – Machine Learning course capstone project**. It brings three core machine learning tasks—classification, regression, and clustering—together in a single web application.

The project demonstrates the end-to-end machine learning workflow, including data preprocessing, model training and comparison, evaluation, and visualization. It provides an interactive interface for exploring datasets, examining model performance, and understanding prediction results.

**Try the application:** [Vectora — Live Demo](https://vectora-m8jqutzw8xnv6njwgymrzu.streamlit.app/)

## Project Tracks

### 1. Star Classification

Classifies astronomical observations from the Sloan Digital Sky Survey (SDSS) into three categories:

- Galaxy
- Star
- Quasi-Stellar Object (QSO)

**Key Features**
- Interactive classification interface.
- Comparison of trained classification models.
- Evaluation using accuracy, weighted F1-score, and ROC-AUC.
- Visualizations of class distributions and model performance.

**Dataset:** SDSS Star Classification Dataset.

### 2. Community Crime Rate Regression

Uses community-level socioeconomic and demographic attributes to estimate violent crime rates.

**Key Features**
- Multiple regression algorithms and model comparison.
- Evaluation using R², Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
- Visualizations of target distributions and feature relationships.
- Interactive regression predictions.

**Dataset:** Communities and Crime Dataset.

### 3. Credit Card Customer Clustering

Applies unsupervised learning to identify customer segments based on credit card usage and account attributes.

**Key Features**
- K-Means clustering with four clusters in the saved segmentation workflow.
- Interactive clustering explorer.
- Cluster visualization and customer segmentation analysis.
- Evaluation using the Silhouette Score and Davies–Bouldin Index.
- Cluster-size analysis and visual exploration.

**Dataset:** Credit Card Customer Dataset (`CC GENERAL.csv`).

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Plotly |
| Model Persistence | Joblib |
| Configuration and Metadata | CSV, JSON |

## Project Structure

```text
Vectora/
├── app.py
├── requirements.txt
├── requirements_gui.txt
├── save_artifacts.py
├── save_models.py
├── ML_CLASSIFICATION_TRACKA/
├── Regression Track A/
└── Clustering/
    ├── CC GENERAL.csv
    ├── app/
    │   └── clustering_preprocessing.py
    └── models/
        ├── cluster_meta.json
        ├── cluster_preprocessor.pkl
        ├── kmeans_k4.pkl
        ├── kmeans_k4_centroids.csv
        └── kmeans_k4_labels.csv
```

*The structure above highlights the main project components. Refer to the repository for the complete list of datasets and model artifacts.*

## Getting Started

### Prerequisites

- Python 3
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Reshma925/Vectora.git
cd Vectora
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If additional GUI dependencies are maintained separately, install them using:

```bash
pip install -r requirements_gui.txt
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

`http://localhost:8501`

## Model Evaluation

| Track | Evaluation Metrics |
|---|---|
| Classification | Accuracy, Weighted F1-score, ROC-AUC |
| Regression | R², RMSE, MAE |
| Clustering | Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index |

These metrics provide different perspectives on classification quality, regression accuracy, and cluster separation.

## Deployment

Vectora is deployed using Streamlit Community Cloud.

**Live Application:** [Open Vectora](https://vectora-m8jqutzw8xnv6njwgymrzu.streamlit.app/)

To deploy another version, push the project to GitHub, ensure the required datasets and model artifacts are available, and configure Streamlit Community Cloud to run the root `app.py` file.

## Course Context

This project was developed for the **23CSE301 – Machine Learning capstone project** during the 2026–2027 academic year. It brings together supervised and unsupervised learning workflows in an interactive application, complementing the implementation and evaluation of machine learning algorithms.

## Future Enhancements

- Additional model comparison and interpretability tools.
- More detailed cluster profiling and visualizations.
- Improved input validation and error handling.
- Expanded automated testing and reproducibility.
