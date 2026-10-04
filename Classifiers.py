"""Classifier comparison with and without PCA dimensionality reduction.

Trains LDA, Decision Tree, k-NN (k = 1, 3, 5, 10) and a linear soft-margin SVM
on TrainingData.csv, evaluates each on TestingData.csv, and reports Type 1 and
Type 2 error rates. The same classifiers are then retrained on PCA-reduced
features. For LDA, Type 1 / Type 2 error rates are plotted as the decision
threshold varies.

Author: Siddhardh Chochipatla

Usage:
    python classifiers.py
    python classifiers.py --train TrainingData.csv --test TestingData.csv
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # save figures to file; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

K_VALUES = [1, 3, 5, 10]
PCA_COMPONENTS = [5, 10, 15, 20]


def load_data(train_path, test_path):
    """Load CSVs; the last column is the label, all others are features."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    return X_train, y_train, X_test, y_test


def calculate_error_rates(y_true, y_pred):
    """Return (type_1_error_rate, type_2_error_rate) for binary labels 0/1.

    Type 1 = FP / (FP + TN): class 0 samples predicted as class 1.
    Type 2 = FN / (FN + TP): class 1 samples predicted as class 0.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    type_1 = fp / (fp + tn) if (fp + tn) else 0.0
    type_2 = fn / (fn + tp) if (fn + tp) else 0.0
    return type_1, type_2


def make_classifiers():
    """Fresh, unfitted classifiers keyed by display name."""
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(),
    }
    for k in K_VALUES:
        models[f"kNN (k={k})"] = KNeighborsClassifier(n_neighbors=k)
    # Soft-margin linear SVM (data is not linearly separable)
    models["SVM"] = LinearSVC(C=0.1, max_iter=1000, dual=False)
    return models


def evaluate(models, X_train, y_train, X_test, y_test, suffix=""):
    """Fit each model, print its error rates, and return them in a dict."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        t1, t2 = calculate_error_rates(y_test, model.predict(X_test))
        results[name] = (t1, t2)
        print(f"{name}{suffix} - Type 1 Error Rate: {t1:.3f}, "
              f"Type 2 Error Rate: {t2:.3f}")
    return results


def lda_threshold_sweep(X_train, y_train, X_test, y_test, out_path):
    """Plot LDA Type 1 / Type 2 error rates as the threshold varies."""
    lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
    scores = lda.decision_function(X_test)
    thresholds = np.linspace(scores.min(), scores.max(), 100)

    type_1_errors, type_2_errors = [], []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        t1, t2 = calculate_error_rates(y_test, preds)
        type_1_errors.append(t1)
        type_2_errors.append(t2)

    plt.figure()
    plt.plot(thresholds, type_1_errors, label="Type 1 Error Rate")
    plt.plot(thresholds, type_2_errors, label="Type 2 Error Rate")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("Type 1 and Type 2 Error Rates vs Threshold (LDA)")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", default="TrainingData.csv")
    parser.add_argument("--test", default="TestingData.csv")
    parser.add_argument("--plot", default="lda_threshold_errors.png")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.train, args.test)

    print("=== Original features ===")
    lda_threshold_sweep(X_train, y_train, X_test, y_test, args.plot)
    evaluate(make_classifiers(), X_train, y_train, X_test, y_test)

    for n in PCA_COMPONENTS:
        print(f"\n=== PCA with {n} components ===")
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        evaluate(make_classifiers(), X_train_pca, y_train, X_test_pca,
                 y_test, suffix=f" with {n} PCA Components")


if __name__ == "__main__":
    main()
