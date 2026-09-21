# Vectora — Star Classification Track A

## 1. Project Overview

This project implements a complete **multiclass classification pipeline** for classifying astronomical objects into three classes:

- **STAR**
- **GALAXY**
- **QSO**

The classification workflow covers dataset loading and auditing, exploratory data analysis (EDA), data cleaning, target encoding, stratified train-test splitting, feature scaling, classification model training, hyperparameter tuning, evaluation, model comparison, and 5-fold cross-validation of the two best-performing models.

The same preprocessed training and testing datasets are used across all five classification algorithms to make the model comparison fair.

---

## 2. Dataset

The notebook loads the dataset from:

```text
star_classification.csv
```

### Dataset size

- **Samples:** 100,000
- **Columns:** 18
- **Input features:** 17
- **Target column:** `class`
- **Target classes:** 3

### Target class distribution

| Class | Samples | Percentage |
|---|---:|---:|
| GALAXY | 59,445 | 59.44% |
| STAR | 21,594 | 21.59% |
| QSO | 18,961 | 18.96% |

The target is therefore multiclass, with some class imbalance. For this reason, **Weighted Precision, Weighted Recall, and Weighted F1-score** are included along with Accuracy.

### Features

The dataset contains the following columns:

```text
obj_ID
alpha
delta
u
g
r
i
z
run_ID
rerun_ID
cam_col
field_ID
spec_obj_ID
redshift
plate
MJD
fiber_ID
```

Target:

```text
class
```

---

## 3. Project Workflow

```text
Dataset Loading
       ↓
Dataset Audit
       ↓
Exploratory Data Analysis
       ↓
Data Cleaning
       ↓
Target / Feature Separation
       ↓
Label Encoding
       ↓
80:20 Stratified Train-Test Split
       ↓
Standard Scaling
       ↓
Train 5 Classification Models
       ↓
Hyperparameter Tuning where required
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Select Top 2 using Weighted F1
       ↓
5-Fold Stratified Cross-Validation
       ↓
Final Cross-Validation Analysis
```

---

# 4. Notebook Structure

The main notebook is:

```text
classification.ipynb
```

The notebook is organized into the following stages.

---

## Part A1 — Dataset Loading & Audit

The dataset is loaded using Pandas.

The initial audit checks:

- Dataset shape
- Column names
- Data types
- Missing values
- Target column
- Class distribution
- Class distribution percentages

The dataset contains **100,000 records and 18 columns**, with no missing values reported in the initial audit.

---

## Part A2 — Exploratory Data Analysis

EDA is performed to understand the dataset before model training.

### Visualizations include:

- Numerical feature distribution plots
- Target class distribution
- Feature correlation heatmap
- Redshift vs. target class
- U-band magnitude vs. target class

The plots are provided with titles, labelled axes, and `tight_layout()` to improve readability.

---

# 5. Data Cleaning

The cleaning stage includes:

### Missing-value checking

The notebook checks every column for missing values.

No missing values were found in the dataset, so no imputation or row deletion was required.

### Duplicate detection

Duplicate rows are checked.

If duplicates are present, they are removed using:

```python
df.drop_duplicates()
```

### Outlier detection

Numerical features are inspected using the **Interquartile Range (IQR)** method.

For each numerical feature:

```text
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

The notebook currently **identifies and reports outliers**; it does not automatically remove or transform them.

---

# 6. Preprocessing

## 6.1 Target Separation

The target column is:

```python
target_column = "class"
```

The dataset is separated into:

```text
X → input features
y → target class
```

---

## 6.2 Label Encoding

The categorical target classes are converted into numerical labels using `LabelEncoder`.

The encoded target is used by the classification algorithms.

---

## 6.3 Stratified Train-Test Split

A consistent **80:20 train-test split** is used for all five models.

```python
test_size = 0.20
random_state = 42
stratify = y_encoded
```

This ensures that approximately the same class proportions are maintained in both the training and testing sets.

Using the same split for all models makes the final metric comparison fair.

---

## 6.4 Feature Scaling

`StandardScaler` is used for feature scaling.

The scaler is:

- fitted only on the training data
- applied to the training data
- reused to transform the test data

This prevents information from the test set from influencing the scaling process.

The scaled datasets are:

```text
X_train_scaled
X_test_scaled
```

---

# 7. Classification Models

Five classification algorithms are implemented.

| # | Model | Main Requirement |
|---:|---|---|
| 1 | Logistic Regression | Baseline classifier; coefficient interpretation |
| 2 | Gaussian Naive Bayes | Conditional independence assumption |
| 3 | K-Nearest Neighbors | Tune `n_neighbors` and distance metric |
| 4 | Decision Tree | Tune `max_depth`; visualize tree |
| 5 | Support Vector Machine | Tune `C` and kernel; use scaled features |

---

## 7.1 Logistic Regression

Logistic Regression is used as the baseline multiclass classifier.

Configuration:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The model coefficients are displayed to examine the relationship between the standardized features and the predicted classes.

Model file:

```text
models/logistic_regression_model.pkl
```

---

## 7.2 Gaussian Naive Bayes

Gaussian Naive Bayes is used as a probabilistic baseline.

The model assumes **conditional independence of the input features given the class**.

In other words, after the class is known, the model treats the features as independent when calculating class probabilities.

For continuous features, Gaussian Naive Bayes also assumes that feature values within each class follow a Gaussian (normal) distribution.

Model file:

```text
models/gaussian_naive_bayes_model.pkl
```

---

## 7.3 K-Nearest Neighbors

KNN classifies observations based on nearby training observations.

The following hyperparameters are tuned:

```python
n_neighbors = [3, 5, 7, 9, 11, 13, 15]

metric = [
    "euclidean",
    "manhattan"
]
```

A 3-fold `GridSearchCV` is used with:

```text
Scoring: F1 Macro
```

### Best KNN configuration

```text
n_neighbors = 3
metric = manhattan
```

Best 3-fold cross-validation F1 Macro:

```text
0.8833
```

Model file:

```text
models/knn_model.pkl
```

---

## 7.4 Decision Tree

A Decision Tree classifier is trained and its maximum depth is tuned.

Hyperparameter search:

```python
max_depth = [
    3,
    5,
    7,
    10,
    15,
    20,
    None
]
```

A 3-fold `GridSearchCV` is used with F1 Macro scoring.

### Best Decision Tree configuration

```text
max_depth = 10
```

Best 3-fold cross-validation F1 Macro:

```text
0.9679
```

The final tree is also visualized using `plot_tree()`.

Model file:

```text
models/decision_tree_model.pkl
```

---

## 7.5 Support Vector Machine

SVM is trained using standardized features.

The following hyperparameters are tuned:

```python
C = [0.1, 1, 10, 100]

kernel = [
    "linear",
    "rbf",
    "poly"
]
```

A 3-fold `GridSearchCV` is used with F1 Macro scoring.

### Best SVM configuration

```text
C = 100
kernel = rbf
```

Best 3-fold cross-validation F1 Macro:

```text
0.9637
```

Model file:

```text
models/svm_model.pkl
```

---

# 8. Model Evaluation

All five models are evaluated using the same test set.

The following metrics are used:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- ROC-AUC (One-vs-Rest)

For the multiclass ROC-AUC calculation, the **One-vs-Rest (OvR)** strategy is used.

---

## 8.1 Final Test-Set Results

The recorded results from the notebook are:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|
| KNN | 0.9076 | 0.9096 | 0.9076 | 0.9064 | 0.9449 |
| Logistic Regression | 0.9574 | 0.9575 | 0.9574 | 0.9570 | 0.9862 |
| Gaussian Naive Bayes | 0.7132 | 0.7704 | 0.7132 | 0.6694 | 0.8975 |
| Decision Tree | 0.9764 | 0.9763 | 0.9764 | 0.9763 | 0.9858 |
| SVM | 0.9719 | 0.9719 | 0.9719 | 0.9717 | 0.9901 |

---

# 9. Model Comparison

The notebook uses **Weighted F1-score as the primary model-selection criterion**.

This metric was selected because it combines precision and recall while accounting for class support.

The five models are ranked using their test-set Weighted F1-score.

The two models selected for additional cross-validation are:

1. Decision Tree
2. SVM

The selection is based on the Weighted F1 ranking from the recorded test-set results.

---

# 10. Five-Fold Cross-Validation

A separate **5-fold Stratified Cross-Validation** evaluation is performed for the two selected models.

Configuration:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Cross-validation is performed using the training data only. The final test set remains separate from this CV procedure.

The scoring metric is:

```text
Weighted F1
```

---

## 10.1 Decision Tree — 5-Fold CV

Fold scores:

```text
[0.9739, 0.9743, 0.9714, 0.9748, 0.9745]
```

Mean Weighted F1:

```text
0.9738
```

Standard deviation:

```text
0.0012
```

---

## 10.2 SVM — 5-Fold CV

Fold scores:

```text
[0.9704, 0.9686, 0.9665, 0.9718, 0.9691]
```

Mean Weighted F1:

```text
0.9693
```

Standard deviation:

```text
0.0018
```

---

# 11. Cross-Validation Conclusion

Based on the recorded 5-fold stratified cross-validation results, the Decision Tree achieved a mean Weighted F1-score of **0.9738** with a standard deviation of **0.0012**, while SVM achieved a mean Weighted F1-score of **0.9693** with a standard deviation of **0.0018**.

Both models maintained consistent performance across the five folds. The Decision Tree obtained the higher mean cross-validation Weighted F1-score in this evaluation.

---

# 12. Visualizations

The notebook includes visualizations for both exploratory analysis and model evaluation.

### EDA visualizations

- Feature distribution plots
- Target class distribution
- Feature correlation heatmap
- Redshift vs. target class
- U-band magnitude vs. target class

### Model visualizations

- Confusion matrix for each classifier
- Decision Tree visualization
- Model metric comparison graphs
- Five-fold cross-validation performance graph
- Mean five-fold Weighted F1 comparison

Plots use titles, labelled axes, legends where applicable, and `tight_layout()` to reduce label clipping.

---

# 13. Saved Models

The trained models are saved using `joblib`.

Expected model directory:

```text
models/
├── knn_model.pkl
├── logistic_regression_model.pkl
├── gaussian_naive_bayes_model.pkl
├── decision_tree_model.pkl
└── svm_model.pkl
```

These files contain the trained model objects selected during the training process.

---

# 14. Repository Structure

The classification track follows this structure:

```text
Vectora/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── star_classification.csv
│
├── ML_CLASSIFICATION_TRACKA/
│   ├── classification.ipynb
│   ├── star_classification.csv
│   └── models/
│       ├── knn_model.pkl
│       ├── logistic_regression_model.pkl
│       ├── gaussian_naive_bayes_model.pkl
│       ├── decision_tree_model.pkl
│       └── svm_model.pkl
│
└── ...
```

The exact location of the dataset/model directory should match the repository structure used in the project.

---

# 15. Requirements

The notebook uses Python and the following main libraries:

```text
pandas
matplotlib
seaborn
scikit-learn
joblib
```

The project also uses:

- `LabelEncoder`
- `StandardScaler`
- `GridSearchCV`
- `StratifiedKFold`
- `cross_val_score`
- `KNeighborsClassifier`
- `LogisticRegression`
- `GaussianNB`
- `DecisionTreeClassifier`
- `SVC`

Install the required dependencies with:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib
```

---

# 16. Running the Notebook

1. Clone the repository.
2. Open the classification track directory.
3. Make sure `star_classification.csv` is available at the path expected by the notebook.
4. Install the required Python dependencies.
5. Open:

```text
classification.ipynb
```

6. Run the notebook cells in order.

The notebook performs:

```text
Loading
→ Audit
→ EDA
→ Cleaning
→ Preprocessing
→ Model Training
→ Evaluation
→ Model Comparison
→ 5-Fold CV
```

---

# 17. Reproducibility

The following settings are fixed to support reproducible experiments:

```text
Train-test split: 80:20
Random state: 42
Stratified split: Yes
5-fold CV shuffle: Yes
5-fold CV random state: 42
```

The same train/test split is used across all five classification algorithms.

---

# 18. Key Findings

From the recorded test-set results:

- Decision Tree achieved a Weighted F1-score of **0.9763**.
- SVM achieved a Weighted F1-score of **0.9717**.
- Logistic Regression achieved a Weighted F1-score of **0.9570**.
- KNN achieved a Weighted F1-score of **0.9064**.
- Gaussian Naive Bayes achieved a Weighted F1-score of **0.6694**.

For the additional 5-fold cross-validation:

- Decision Tree mean Weighted F1: **0.9738**
- SVM mean Weighted F1: **0.9693**

The cross-validation results show that both selected models maintain consistent performance across folds, with the Decision Tree having the higher mean Weighted F1 in this evaluation.

---

## 19. Status

### Completed

- [x] Dataset loading and audit
- [x] Missing-value checking
- [x] Duplicate checking
- [x] IQR-based outlier identification
- [x] Exploratory data analysis
- [x] Target encoding
- [x] Stratified 80:20 train-test split
- [x] Standard feature scaling
- [x] Logistic Regression
- [x] Gaussian Naive Bayes
- [x] KNN with GridSearchCV
- [x] Decision Tree with GridSearchCV
- [x] SVM with GridSearchCV
- [x] Model evaluation
- [x] Confusion matrices
- [x] Model comparison
- [x] Weighted F1-based ranking
- [x] 5-fold cross-validation for the top two models
- [x] Cross-validation visualizations
- [x] Model saving using Joblib

### Pending / Extension

- [ ] Add and evaluate a justified engineered feature
- [ ] Add further deployment/application components if required
