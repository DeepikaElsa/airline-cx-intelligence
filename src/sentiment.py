"""TF-IDF + Logistic Regression sentiment model with probability calibration."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from src.config import CV_FOLDS, RANDOM_STATE
from src.preprocess import LABELS


def build_model():
    base = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced"),
    )
    # Sigmoid (Platt) calibration makes the probabilities more trustworthy
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def out_of_fold_probabilities(df):
    """Every tweet gets a probability predicted by a model that never saw it.
    Columns follow LABELS: negative, neutral, positive."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(build_model(), df["clean_text"], df["airline_sentiment"],
                             cv=cv, method="predict_proba")


def evaluate(y_true, proba):
    y_true = np.asarray(y_true)
    pred = np.array(LABELS)[proba.argmax(axis=1)]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=LABELS, zero_division=0)
    per_class = {
        lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for lab, p, r, f, s in zip(LABELS, precision, recall, f1, support)
    }

    neg_true = (y_true == "negative").astype(int)
    frac_pos, mean_pred = calibration_curve(neg_true, proba[:, 0], n_bins=10, strategy="uniform")

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS,
                             "matrix": confusion_matrix(y_true, pred, labels=LABELS).tolist()},
        "calibration": {
            "brier_negative": float(brier_score_loss(neg_true, proba[:, 0])),
            "mean_predicted": [float(x) for x in mean_pred],
            "observed_rate": [float(x) for x in frac_pos],
        },
    }
