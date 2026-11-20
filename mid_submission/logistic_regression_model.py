# discriminative L2-regularized logistic regression classifier pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class LogisticRegressionModel:
    # initialize dict vectorizer and L2 logistic regression classifier
    def __init__(self, max_iter=1000):
        self.model = Pipeline([("vectorizer", DictVectorizer()),("classifier", LogisticRegression(max_iter=max_iter,random_state=42))])

    # fit classifier on training feature dicts and sentiment labels
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # predict binary class labels (0 = negative, 1 = positive)
    def predict(self, X):
        return self.model.predict(X)

    # predict class probability distributions
    def predict_proba(self, X):
        return self.model.predict_proba(X)
