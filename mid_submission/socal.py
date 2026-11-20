# rule-based SO-CAL sentiment scorer using lexicon polarity, clause-bounded negation, and intensifiers
import numpy as np
from lexicon_semantics import build_lexicon
from preprocess import preprocess, flatten
from config import INTENSIFIERS

# negation words that flip token polarity within the same clause
NEGATORS = {"not", "n't", "never", "no", "hardly"}

class SOCal:
    # load sentiment lexicon if not already provided
    def __init__(self, lexicon=None):
        self.lexicon = lexicon if lexicon is not None else build_lexicon()

    # compute average shifted polarity score for a single review
    def score_review(self, review):
        sentences = preprocess(review)
        tokens = flatten(sentences)
        scores = []

        for i, token in enumerate(tokens):
            # improvement 2b: Check lowercase word first, then fallback to lemma
            polarity = self.lexicon.get(token.text.lower(), self.lexicon.get(token.lemma, 0.0))

            if polarity == 0:
                continue

            # Check preceding tokens for negation
            for j in range(max(0, i - 6), i):
                previous = tokens[j]
                # improvement 3: Do not allow negation to cross sentence or clause boundaries
                if previous.sent_id != token.sent_id or previous.clause_id != token.clause_id:
                    continue

                if previous.text.lower() in NEGATORS:
                    polarity = -polarity
                    break

            # Check preceding tokens for intensifiers/downtoners
            for j in range(max(0, i - 3), i):
                previous = tokens[j]
                # improvement 3: Do not allow intensifiers to cross sentence or clause boundaries
                if previous.sent_id != token.sent_id or previous.clause_id != token.clause_id:
                    continue
                word = previous.text.lower()

                if word in INTENSIFIERS:
                    polarity *= INTENSIFIERS[word]

            scores.append(polarity)

        if not scores:
            return 0.0

        # return mean polarity across all opinion tokens in the review
        return float(np.mean(scores))

    # classify reviews as positive (1), negative (0), or neutral (-1)
    def predict(self, reviews):
        scores = self.predict_scores(reviews)
        return np.where(scores > 0, 1, np.where(scores < 0, 0, -1))

    # compute array of raw SO-CAL scores for a list of reviews
    def predict_scores(self, reviews):
        return np.array([self.score_review(review) for review in reviews])