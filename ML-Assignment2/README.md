# Dry Bean Variety Classification – ML Assignment 2

## a. Problem statement

Dry beans are sorted into commercial varieties before they are sold, and sorting by hand is slow and inconsistent. The goal of this project is to predict the variety of a dry bean grain (7 classes) from 16 shape measurements taken from an image of the grain. Five classification models are trained on the same data, compared on six evaluation metrics, and served through a Streamlit web app where a user can upload test data, pick a model, and see its metrics, confusion matrix and classification report.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/602/dry+bean+dataset (Koklu & Ozkan, 2020) |
| Task | Multi-class classification (7 classes) |
| Instances | 13,611 (13,543 after removing 68 duplicate rows) |
| Features | 16 numeric features (12 dimension features + 4 shape factors) |
| Target column | `Class` |
| Missing values | None |

**Features:** `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`.

**Class distribution (after de-duplication):** DERMASON 3546, SIRA 2636, SEKER 2027, HOROZ 1860, CALI 1630, BARBUNYA 1322, BOMBAY 522. The classes are imbalanced (BOMBAY is about 4% of the data), so macro-averaged Precision, Recall and F1 are reported, which weight every class equally.

**Preprocessing:**
- Duplicate rows removed.
- Stratified 80/20 train/test split with `random_state=42` (10,834 train rows, 2,709 test rows). The test split is saved as `test_data.csv`.
- `StandardScaler` is applied inside a scikit-learn `Pipeline` for Logistic Regression, KNN, Naive Bayes and SVM, so the saved model files already contain the scaler. The Decision Tree uses raw features because trees are not affected by feature scale.

## c. GitHub repository link

https://github.com/<your-username>/<your-repo-name>

**Live Streamlit app:** https://drybean-gayathri.streamlit.app/

### Repository structure

```
project-folder/
├── app.py                 # Streamlit app (loads saved models, no retraining)
├── train_models.py        # trains the 5 models, saves them and the test split
├── metrics_utils.py       # feature list + metric calculation shared by both scripts
├── requirements.txt
├── README.md
├── test_data.csv          # 20% held-out test split (16 features + Class)
├── data/
│   └── dry_bean_full.csv  # full dataset converted from the UCI .xlsx file
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metrics.json       # test-set metrics recorded at training time
└── .streamlit/config.toml # app theme
```

## d. Models used

| # | Model | Configuration |
|---|---|---|
| 1 | Logistic Regression | StandardScaler + `LogisticRegression(max_iter=3000, C=1.0)` (multinomial) |
| 2 | Decision Tree | `DecisionTreeClassifier(max_depth=12, min_samples_leaf=3)` |
| 3 | K-Nearest Neighbors | StandardScaler + `KNeighborsClassifier(n_neighbors=9, weights="distance")` |
| 4 | Naive Bayes | StandardScaler + `GaussianNB()` |
| 5 | Support Vector Machine | StandardScaler + `SVC(kernel="rbf", C=10, gamma="scale", probability=True)` |

### Comparison table (test set, 2,709 rows)

Precision, Recall and F1 are macro-averaged. AUC is the macro one-vs-rest ROC AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.9040 | 0.9700 | 0.9179 | 0.9179 | 0.9178 | 0.8840 |
| KNN | 0.9133 | 0.9872 | 0.9277 | 0.9239 | 0.9254 | 0.8951 |
| Naive Bayes | 0.8970 | 0.9911 | 0.9081 | 0.9082 | 0.9072 | 0.8762 |
| Support Vector Machine | **0.9236** | 0.9944 | **0.9356** | **0.9340** | **0.9347** | **0.9076** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall (accuracy 0.919, MCC 0.902) and the highest AUC (0.9947). After scaling, most classes are close to linearly separable in this feature space, so a linear model already does very well. Its main errors are between DERMASON and SIRA (94 test rows confused in either direction), two varieties with very similar size and shape. |
| Decision Tree | Lowest AUC (0.970) and second-lowest accuracy (0.904). A single tree gives step-like probability estimates, which hurts AUC. It also had the most DERMASON/SIRA mix-ups (113) and the lowest SIRA F1 (0.834). Limiting depth to 12 and leaf size to 3 reduced overfitting, but one tree is still less stable than the other models. |
| KNN | Mid-table (accuracy 0.913, MCC 0.895). It works well because beans of the same variety form compact clusters once features are scaled, but it is slightly behind LR and SVM on the overlapping DERMASON/SIRA and BARBUNYA/CALI pairs. It does no real training but stores the whole training set, so it is the largest model file (about 1.5 MB) and every prediction needs a distance search over 10,834 rows. |
| Naive Bayes | Lowest accuracy (0.897) and MCC (0.876), but a high AUC (0.991). Many features are strongly correlated (Area, Perimeter, ConvexArea, EquivDiameter), which breaks the independence assumption. The class ranking stays good but the hard decisions are worse, especially BARBUNYA predicted as CALI (40 rows, more than double the other models) and BARBUNYA F1 of 0.832. |
| Support Vector Machine | Best on accuracy (0.924), precision, recall, F1 (0.935) and MCC (0.908), with AUC (0.9944) almost equal to LR. The RBF kernel draws non-linear boundaries, which helps most on the overlapping classes: it has the best F1 on BARBUNYA, CALI, DERMASON, SEKER and SIRA. It is the slowest model to train and predict (about 1.5 s to fit versus 0.1 s or less for the others) because of the kernel computations and the internal cross-validation used to produce probabilities. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF).** It leads on 5 of the 6 metrics, and Logistic Regression is a very close second. All five models classify BOMBAY perfectly (F1 = 1.0) because it is much larger than the other beans. Every model's main remaining error is DERMASON vs SIRA. |

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_models.py           # trains models, writes model/ and test_data.csv
streamlit run app.py
```

## Streamlit app features

- CSV upload of test data (sidebar), plus a button to download the sample `test_data.csv`.
- Dropdown to choose any of the 5 trained models.
- Display of Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- Confusion matrix heatmap and per-class classification report.
- Comparison tab that evaluates all 5 models on the uploaded file, with a bar chart.
- Prediction table with a filter for misclassified rows and a CSV download.

The app only loads the saved `.joblib` files and calls `predict` / `predict_proba`. The models are never retrained on the uploaded data.
