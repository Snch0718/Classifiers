# Classifier Comparison with PCA

Compares four classifiers on a binary classification dataset (Approved vs. Denied), both on the original features and on PCA-reduced features. Performance is reported as Type 1 and Type 2 error rates on a held-out test set.

**Author:** Siddhardh Chochipatla (UID 121093646)

## Classifiers

| Classifier | Notes |
|---|---|
| Linear Discriminant Analysis (LDA) | Finds the projection that maximizes between-class vs. within-class variance. Type 1 / Type 2 error rates are also plotted as the decision threshold varies. |
| Decision Tree | Recursively partitions the feature space (Gini impurity). |
| k-Nearest Neighbors | Evaluated for k = 1, 3, 5, 10. |
| Linear SVM | Soft-margin (`C=0.1`) since the data is not linearly separable. |

## Dimensionality reduction

PCA is applied with 5, 10, 15 and 20 components (fit on the training set, applied to the test set), and every classifier is retrained and re-evaluated on the reduced features. Edit `PCA_COMPONENTS` in `classifiers.py` to change this.

## Error definitions

For binary labels 0 and 1:

- **Type 1 error rate** = FP / (FP + TN), i.e. class 0 samples predicted as class 1.
- **Type 2 error rate** = FN / (FN + TP), i.e. class 1 samples predicted as class 0.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place `TrainingData.csv` and `TestingData.csv` in the same directory as `classifiers.py`. In each file, the last column is the label (0/1) and all other columns are features. The data files are not included in this repo.

## Run

```bash
python classifiers.py
# or with custom paths
python classifiers.py --train path/to/TrainingData.csv --test path/to/TestingData.csv --plot out.png
```

Output:

- Type 1 / Type 2 error rates printed to the console for every classifier, first on the original features, then for each PCA setting.
- `lda_threshold_errors.png`: LDA Type 1 and Type 2 error rates vs. threshold.

## Files

- `classifiers.py`: full pipeline
- `requirements.txt`: Python dependencies
- `README.md`: this file