# ML Classification Track A -- Star Classification

## Overview

This project implements a complete machine learning classification
pipeline for classifying astronomical observations using multiple
supervised machine learning algorithms.

The workflow covers:

-   Dataset loading and auditing
-   Exploratory Data Analysis (EDA)
-   Distribution analysis
-   Correlation analysis
-   Target-vs-feature visualization
-   Feature engineering
-   Missing-value checking
-   Duplicate detection and handling
-   Outlier detection and IQR-based capping
-   Stratified train-test splitting
-   Target encoding
-   Feature scaling
-   Hyperparameter tuning
-   Training of five classification models
-   Model evaluation using multiple metrics
-   Confusion matrix visualization
-   Model-specific visualizations
-   Model performance comparison
-   Saving trained models for reuse

------------------------------------------------------------------------

## 1. Project Objective

The objective of this project is to develop and compare multiple machine
learning classification models for a multiclass astronomical
classification problem.

Five classification algorithms are implemented:

1.  Logistic Regression
2.  Gaussian Naive Bayes
3.  K-Nearest Neighbors (KNN)
4.  Decision Tree
5.  Support Vector Machine (SVM)

The models are evaluated using a common train-test split and multiple
classification metrics.

------------------------------------------------------------------------

## 2. Dataset

The project uses:

``` text
star_classification.csv
```

The target variable is:

``` text
class
```

The dataset contains astronomical observation features including:

``` text
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

A derived feature named `r_i` is also created during feature
engineering.

------------------------------------------------------------------------

## 3. Technologies and Libraries

The project is implemented in Python using:

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Joblib

Main Scikit-learn components include:

-   `train_test_split`
-   `StandardScaler`
-   `LabelEncoder`
-   `GridSearchCV`
-   `LogisticRegression`
-   `GaussianNB`
-   `KNeighborsClassifier`
-   `DecisionTreeClassifier`
-   `SVC`
-   Classification metrics
-   Confusion matrix
-   ROC-AUC

------------------------------------------------------------------------

## 4. Project Workflow

``` text
Dataset
   ↓
Dataset Audit
   ↓
Exploratory Data Analysis
   ↓
Correlation Analysis
   ↓
Feature Engineering
   ↓
Missing Value Check
   ↓
Duplicate Detection / Handling
   ↓
Separate Features and Target
   ↓
Stratified Train-Test Split
   ↓
Outlier Detection
   ↓
IQR-Based Outlier Capping
   ↓
Target Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Save Trained Models
```

------------------------------------------------------------------------

## 5. Dataset Loading and Audit

The dataset is loaded using Pandas:

``` python
df = pd.read_csv("star_classification.csv")
```

The initial audit checks:

-   Dataset shape
-   Column names
-   Data types
-   Missing values
-   Target column
-   Class distribution
-   Class distribution percentages

This provides an initial understanding of the dataset before
preprocessing.

------------------------------------------------------------------------

## 6. Exploratory Data Analysis

### 6.1 Feature Distribution Analysis

Distribution plots are generated for numerical features using histograms
with KDE.

The plots are arranged in a grid layout for easier visualization.

They help identify:

-   Feature ranges
-   Skewness
-   Concentration of observations
-   Potential extreme values
-   General feature behaviour

### 6.2 Target Class Distribution

A count plot is used to visualize the distribution of the target
classes.

This helps examine the number of observations belonging to each class.

### 6.3 Correlation Analysis

A correlation matrix is calculated for the numerical features.

The correlation heatmap helps identify:

-   Strong positive relationships
-   Strong negative relationships
-   Weak relationships
-   Highly related features

------------------------------------------------------------------------

## 7. Target vs Feature Visualization

Scatter plots are generated between numerical features and the target
classes.

The plots are arranged in a grid layout to allow comparison across all
features.

These plots provide a visual understanding of how the feature values
vary across target classes.

------------------------------------------------------------------------

## 8. Feature Engineering

A new engineered feature is created:

``` python
df["r_i"] = df["r"] - df["i"]
```

The feature is defined as:

``` text
r_i = r - i
```

### Why `r_i`?

The `r` and `i` variables represent measurements in different
photometric bands.

Their difference forms an astronomical colour index:

``` text
r - i
```

This captures the relative difference between the two measurements and
can provide additional information about the object's colour/spectral
characteristics.

The engineered feature is created before the train-test split, so it
becomes part of the dataset and is subsequently available to both
training and testing feature sets.

------------------------------------------------------------------------

## 9. Data Cleaning

### 9.1 Missing Values

The notebook checks every column for missing values.

If missing values are present, their counts are displayed. If no missing
values are present, no unnecessary imputation is performed.

### 9.2 Duplicate Records

Duplicate rows are detected using:

``` python
df.duplicated()
```

If duplicates are present, they are removed using:

``` python
df.drop_duplicates()
```

The resulting dataset shape is then verified.

------------------------------------------------------------------------

## 10. Train-Test Split

The target column is:

``` text
class
```

The dataset is separated into:

``` text
X → Features
y → Target
```

An 80:20 stratified train-test split is used:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Why stratification?

Stratification maintains approximately the same class proportions in the
training and testing datasets.

The random state is fixed at:

``` text
42
```

to make the split reproducible.

------------------------------------------------------------------------

## 11. Outlier Detection and Handling

Outliers are detected for all numerical features in the training dataset
using the Interquartile Range (IQR) method.

The IQR is:

``` text
IQR = Q3 - Q1
```

The boundaries are:

``` text
Lower Boundary = Q1 - 1.5 × IQR

Upper Boundary = Q3 + 1.5 × IQR
```

Values outside these boundaries are identified as statistical outliers.

The notebook produces an outlier summary containing:

-   Feature
-   Q1
-   Q3
-   IQR
-   Lower Boundary
-   Upper Boundary
-   Lower Outliers
-   Upper Outliers
-   Total Outliers

### 11.1 IQR Capping

Outliers are handled using capping rather than removing rows.

For each affected feature:

``` python
X_train[feature] = X_train[feature].clip(
    lower=lower_bound,
    upper=upper_bound
)
```

The important preprocessing rule is that the IQR boundaries are
calculated from the training data only.

The same training-derived boundaries are then applied to the test data.

``` text
Training data
     ↓
Calculate Q1 and Q3
     ↓
Calculate IQR boundaries
     ↓
Cap training values
     ↓
Apply SAME boundaries to test data
```

No rows are removed during outlier handling.

The notebook also verifies the number of outliers before and after
capping.

------------------------------------------------------------------------

## 12. Target Encoding

The categorical target variable is converted into numerical class labels
using `LabelEncoder`.

The encoder is fitted only on the training target:

``` python
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
```

The same fitted encoder is then applied to the test target:

``` python
y_test = label_encoder.transform(y_test)
```

This keeps the class mapping consistent between training and testing
data.

------------------------------------------------------------------------

## 13. Feature Scaling

Feature scaling is performed using `StandardScaler`.

The scaler is fitted only on the training features:

``` python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
```

The same fitted scaler is applied to the test features:

``` python
X_test_scaled = scaler.transform(X_test)
```

This prevents information from the test set from being used to calculate
scaling parameters.

------------------------------------------------------------------------

## 14. Classification Models

Five machine learning classification algorithms are implemented.

### 14.1 Logistic Regression

Logistic Regression is used as a multiclass classification model.

The trained model is saved for later reuse.

### 14.2 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm that
assumes conditional independence between features given the class.

### 14.3 K-Nearest Neighbors

KNN classifies observations based on nearby training observations.

Hyperparameter tuning is performed using `GridSearchCV`.

Parameters include:

``` text
n_neighbors:
3, 5, 7, 9, 11, 13, 15

metric:
euclidean
manhattan
```

The tuning uses cross-validation and macro F1 scoring.

### 14.4 Decision Tree

A Decision Tree classifier is tuned using `GridSearchCV`.

The `max_depth` values tested include:

``` text
3
5
7
10
15
20
None
```

The tuned model is then evaluated on the test set.

A visualization of the trained Decision Tree is also generated.

### 14.5 Support Vector Machine

SVM is implemented using `SVC`.

Probability estimation is enabled so that ROC-AUC can be calculated.

The hyperparameters include:

``` text
C:
0.1, 1, 10, 100

kernel:
linear, rbf, poly
```

The tuned model is evaluated on the test set.

------------------------------------------------------------------------

## 15. Hyperparameter Tuning

`GridSearchCV` is used for hyperparameter tuning of selected models.

The general procedure is:

``` text
Define Model
     ↓
Define Parameter Grid
     ↓
GridSearchCV
     ↓
Cross-Validation
     ↓
Evaluate Parameter Combinations
     ↓
Select Best Configuration
     ↓
Train Best Estimator
```

The tuning objective uses F1-based scoring so that classification
performance across classes is considered during parameter selection.

------------------------------------------------------------------------

## 16. Model Evaluation

All five models are evaluated on the same test dataset.

The following metrics are calculated:

### Accuracy

Measures the proportion of correctly classified observations.

``` text
Accuracy =
Correct Predictions / Total Predictions
```

### Weighted Precision

Precision is calculated for each class and weighted according to class
support.

### Weighted Recall

Recall is calculated for each class and weighted according to class
support.

### Weighted F1-score

F1-score combines precision and recall using their harmonic mean and
weights each class according to its support.

### ROC-AUC

ROC-AUC is used to evaluate class discrimination.

For the multiclass problem, the One-vs-Rest approach is used where
applicable.

------------------------------------------------------------------------

## 17. Confusion Matrices

A confusion matrix is generated for each classification model.

The matrices show:

-   Actual classes
-   Predicted classes
-   Correct classifications
-   Misclassifications

They are visualized using Seaborn heatmaps.

------------------------------------------------------------------------

## 18. Model-Specific Visualizations

### Logistic Regression

A coefficient table is generated to inspect the learned feature
coefficients.

### Decision Tree

The trained Decision Tree is visualized using `plot_tree()`.

### All Models

Confusion matrices are generated for:

-   Logistic Regression
-   Gaussian Naive Bayes
-   KNN
-   Decision Tree
-   SVM

------------------------------------------------------------------------

## 19. Cross-Validation

Cross-validation can be used to examine how consistently a model
performs across different subsets of the training data.

For example, the final validation section can report fold-level F1
scores, their mean, and their standard deviation.

A representative output format is:

``` text
Decision Tree - 5-Fold Cross-Validation
--------------------------------------
Fold Scores: [ ... ]
Mean Weighted F1: ...
Standard Deviation: ...

SVM - 5-Fold Cross-Validation
--------------------------------------
Fold Scores: [ ... ]
Mean Weighted F1: ...
Standard Deviation: ...
```

The mean score summarizes average performance across folds, while the
standard deviation indicates the variation between folds.

------------------------------------------------------------------------

## 20. Model Performance Comparison

The models are compared using:

-   Accuracy
-   Weighted Precision
-   Weighted Recall
-   Weighted F1
-   ROC-AUC

The results are visualized using metric comparison plots.

The comparison is used to understand the relative performance of the
five trained models across different evaluation criteria.

------------------------------------------------------------------------

## 21. Model Files

The trained models can be stored using Joblib.

The intended model directory is:

``` text
models/
│
├── logistic_regression_model.pkl
├── gaussian_naive_bayes_model.pkl
├── knn_model.pkl
├── decision_tree_model.pkl
└── svm_model.pkl
```

A saved model can later be loaded using:

``` python
import joblib

model = joblib.load("models/decision_tree_model.pkl")
```

------------------------------------------------------------------------

## 22. Project Structure

``` text
Vectora/
│
├── ML_CLASSIFICATION_TRACKA/
│   │
│   ├── classification.ipynb
│   ├── star_classification.csv
│   │
│   └── models/
│       ├── logistic_regression_model.pkl
│       ├── gaussian_naive_bayes_model.pkl
│       ├── knn_model.pkl
│       ├── decision_tree_model.pkl
│       └── svm_model.pkl
│
└── README.md
```

------------------------------------------------------------------------

## 23. Reproducibility

The project uses fixed random states where applicable.

For the train-test split:

``` text
random_state = 42
```

This makes the data split reproducible when the same dataset and
environment are used.

------------------------------------------------------------------------

## 24. How to Run the Project

### Step 1 -- Clone the repository

``` bash
git clone <repository-url>
```

### Step 2 -- Navigate to the project

``` bash
cd Vectora
```

### Step 3 -- Install the required libraries

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 4 -- Open the notebook

Open:

``` text
ML_CLASSIFICATION_TRACKA/classification.ipynb
```

using VS Code, Jupyter Notebook, or JupyterLab.

### Step 5 -- Ensure the dataset is available

Make sure:

``` text
star_classification.csv
```

is available at the path expected by the notebook.

### Step 6 -- Run the notebook

Run the notebook cells from top to bottom.

The notebook performs:

1.  Dataset loading
2.  Dataset auditing
3.  Exploratory data analysis
4.  Feature engineering
5.  Data cleaning
6.  Train-test splitting
7.  Outlier detection and capping
8.  Target encoding
9.  Feature scaling
10. Model training
11. Hyperparameter tuning
12. Model evaluation
13. Visualization
14. Model comparison
15. Model saving

------------------------------------------------------------------------

## 25. Methodology Summary

``` text
                STAR CLASSIFICATION DATASET
                           │
                           ▼
                  Dataset Loading & Audit
                           │
                           ▼
                  Exploratory Data Analysis
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Distributions   Correlation   Target vs
                       Analysis      Features
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Feature Engineering
                           │
                      r_i = r - i
                           │
                           ▼
                     Data Cleaning
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Missing Values         Duplicates
                 │                   │
                 └─────────┬─────────┘
                           ▼
                Stratified Train-Test Split
                        80% / 20%
                           │
                           ▼
                   Outlier Detection
                       IQR Method
                           │
                           ▼
                     IQR Capping
                           │
                           ▼
                    Target Encoding
                           │
                           ▼
                    Feature Scaling
                    StandardScaler
                           │
                           ▼
                  Model Training & Tuning
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Logistic         Naive Bayes          KNN
    Regression
          │                │                │
          └────────────────┼────────────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Decision Tree              SVM
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    Test Predictions
                           │
                           ▼
                    Model Evaluation
                           │
       ┌──────────┬────────┬────────┬──────────┐
       ▼          ▼        ▼        ▼          ▼
    Accuracy  Precision  Recall  F1-score  ROC-AUC
                           │
                           ▼
                  Model Comparison
                           │
                           ▼
                    Saved Models
```

------------------------------------------------------------------------

## 26. Evaluation Metrics Used

  -----------------------------------------------------------------------
  Metric                              Purpose
  ----------------------------------- -----------------------------------
  Accuracy                            Measures overall correct
                                      predictions

  Weighted Precision                  Measures precision while accounting
                                      for class support

  Weighted Recall                     Measures recall while accounting
                                      for class support

  Weighted F1                         Provides a weighted balance between
                                      precision and recall

  ROC-AUC                             Measures multiclass discrimination
                                      using One-vs-Rest
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 27. Key Preprocessing Decisions

  Step                    Method
  ----------------------- ------------------------------------------
  Missing values          Checked before modelling
  Duplicate records       Detected and handled
  Feature engineering     `r_i = r - i`
  Train-test split        80:20 stratified split
  Random state            42
  Outlier detection       IQR method
  Outlier handling        IQR capping
  Target encoding         LabelEncoder
  Feature scaling         StandardScaler
  Hyperparameter tuning   GridSearchCV
  Model evaluation        Accuracy, Precision, Recall, F1, ROC-AUC
  Model persistence       Joblib

------------------------------------------------------------------------

## 28. Conclusion

This project implements an end-to-end multiclass machine learning
classification workflow for astronomical data.

The pipeline combines exploratory data analysis, domain-informed feature
engineering, data cleaning, statistical outlier handling, stratified
splitting, target encoding, feature scaling, hyperparameter tuning,
model training, and multi-metric evaluation.

Five classification algorithms are trained and evaluated under a common
preprocessing and testing framework:

-   Logistic Regression
-   Gaussian Naive Bayes
-   KNN
-   Decision Tree
-   SVM

The project also includes visual analysis through distribution plots,
correlation heatmaps, target-vs-feature scatter plots, confusion
matrices, model-specific plots, and model performance comparisons.

Trained models can be saved using Joblib for future reuse.

------------------------------------------------------------------------

## Author

**Maheswara Manikanta Pullela**

**Machine Learning Classification -- Track A**
