# generative multinomial naive bayes classifier pipeline with laplace smoothing
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class NaiveBayesModel:
    # initialize dict vectorizer and multinomial naive bayes with add-alpha smoothing
    def __init__(self, alpha=1.0):
        self.model = Pipeline([("vectorizer", DictVectorizer()),("classifier", MultinomialNB(alpha=alpha))])

    # improvement 4c: Filter negative features and scale tiny densities (x100) so MultinomialNB uses M1-M4 features
    def _prepare_nb_features(self, X):
        signed_features = {"lex_polarity_sum", "shift_sum", "lex_polarity_density", "shift_density", "ana_salient_net_density", "rst_weighted_polarity_density"}
        return [
            {
                k:(v * 100.0 if k.startswith(("lex_", "shift_")) else (v * 4.0 if k.startswith(("ana_", "rst_")) else v))
                for k, v in row.items()
                if k not in signed_features and v >= 0
            }
            for row in X
        ]

    # fit naive bayes model on non-negative feature dicts
    def train(self, X_train, y_train):
        self.model.fit(self._prepare_nb_features(X_train), y_train)

    # predict binary class labels (0 = negative, 1 = positive)
    def predict(self, X):
        return self.model.predict(self._prepare_nb_features(X))

    # predict class probability distributions
    def predict_proba(self, X):
        return self.model.predict_proba(self._prepare_nb_features(X))
