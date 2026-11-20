# mid-submission evaluation script: trains and evaluates M0, M1, M2, and SO-CAL on train/val/test splits
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lexicon_semantics import build_lexicon
from features import extract_features
from logistic_regression_model import LogisticRegressionModel
from naive_bayes_model import NaiveBayesModel
from socal import SOCal

# read review text and sentiment label (0 or 1) from a csv split file
def load_csv_split(filepath):
    reviews = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append(row["review"])
            labels.append(int(row["sentiment"]))
    return reviews, labels

# extract feature dictionaries for a list of reviews under model M0, M1, or M2
def build_feature_list(reviews, model_name, lexicon):
    feature_list = []
    total = len(reviews)
    for idx, rev in enumerate(reviews):
        if (idx + 1) % 400 == 0 or (idx + 1) == total:
            print(f"  [{model_name}] extracted features for {idx + 1}/{total} reviews")
        feats = extract_features(rev, model=model_name, lexicon=lexicon)
        feature_list.append(feats)
    return feature_list

def main():
    print("=== Loading Dataset Splits from processed_data/ ===")
    train_reviews, y_train = load_csv_split("processed_data/train.csv")
    val_reviews, y_val = load_csv_split("processed_data/validation.csv")
    test_reviews, y_test = load_csv_split("processed_data/test.csv")
    print(f"Loaded Train: {len(train_reviews)}, Validation: {len(val_reviews)}, Test: {len(test_reviews)}\n")

    lexicon = build_lexicon()
    summary_rows = []

    # evaluate M0, M1, and M2 with Logistic Regression and Multinomial Naive Bayes
    for model_name in ["M0", "M1", "M2"]:
        print(f"=== Preparing Features for {model_name} ===")
        X_train = build_feature_list(train_reviews, model_name, lexicon)
        X_val = build_feature_list(val_reviews, model_name, lexicon)
        X_test = build_feature_list(test_reviews, model_name, lexicon)

        version_str = "Raw" if model_name == "M0" else "Normalized"

        # 1. train and evaluate Logistic Regression
        lr = LogisticRegressionModel()
        lr.train(X_train, y_train)
        lr_val_pred = lr.predict(X_val)
        lr_test_pred = lr.predict(X_test)

        summary_rows.append({
            "Classifier": "Logistic Regression",
            "Features": model_name,
            "Version": version_str,
            "Validation Accuracy": round(accuracy_score(y_val, lr_val_pred), 4),
            "Test Accuracy": round(accuracy_score(y_test, lr_test_pred), 4),
            "Precision": round(precision_score(y_test, lr_test_pred), 4),
            "Recall": round(recall_score(y_test, lr_test_pred), 4),
            "F1": round(f1_score(y_test, lr_test_pred), 4)
        })

        # 2. train and evaluate Multinomial Naive Bayes
        nb = NaiveBayesModel()
        nb.train(X_train, y_train)
        nb_val_pred = nb.predict(X_val)
        nb_test_pred = nb.predict(X_test)

        summary_rows.append({
            "Classifier": "Naive Bayes",
            "Features": model_name,
            "Version": version_str,
            "Validation Accuracy": round(accuracy_score(y_val, nb_val_pred), 4),
            "Test Accuracy": round(accuracy_score(y_test, nb_test_pred), 4),
            "Precision": round(precision_score(y_test, nb_test_pred), 4),
            "Recall": round(recall_score(y_test, nb_test_pred), 4),
            "F1": round(f1_score(y_test, nb_test_pred), 4)
        })
        print()

    # 3. evaluate Rule-Based SO-CAL baseline
    print("=== Evaluating Rule-Based SO-CAL Scorer ===")
    socal_model = SOCal(lexicon)
    val_scores = socal_model.predict_scores(val_reviews)
    test_scores = socal_model.predict_scores(test_reviews)

    # tune threshold on validation split
    best_th = -0.10
    best_val_f1 = -1.0
    for step in range(-20, 21):
        th = step * 0.01
        val_p = [1 if s > th else 0 for s in val_scores]
        vf1 = f1_score(y_val, val_p, zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_th = th

    socal_val_pred = [1 if s > best_th else 0 for s in val_scores]
    socal_test_pred = [1 if s > best_th else 0 for s in test_scores]

    summary_rows.append({
        "Classifier": "SO-CAL",
        "Features": "Lexicon-based",
        "Version": f"Threshold = {round(best_th, 2)}",
        "Validation Accuracy": round(accuracy_score(y_val, socal_val_pred), 4),
        "Test Accuracy": round(accuracy_score(y_test, socal_test_pred), 4),
        "Precision": round(precision_score(y_test, socal_test_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, socal_test_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, socal_test_pred, zero_division=0), 4)
    })

    print("\n=== Mid-Submission Evaluation Results (M0, M1, M2, and SO-CAL) ===")
    for r in summary_rows:
        print(
            f"{r['Classifier']:20s} | {r['Features']:14s} | "
            f"Val Acc: {r['Validation Accuracy']:.4f} | "
            f"Test Acc: {r['Test Accuracy']:.4f} | F1: {r['F1']:.4f}"
        )

if __name__ == "__main__":
    main()