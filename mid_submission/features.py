# feature extraction pipeline combining M0 (bag-of-words), M1 (lexicon), and M2 (compositional shifters)
from collections import Counter

from preprocess import preprocess
from lexicon_semantics import apply_lexicon, lexicon_features
from compositional_rules import apply_shifters, shifter_features

# extract unigram (UNI_) and bigram (BI_) frequency counts from preprocessed sentences (model M0)
def bow_features(sentences):
    features = Counter()

    for sentence in sentences:
        # keep only alphanumeric word lemmas
        tokens = [token.lemma.lower() for token in sentence.tokens if any(ch.isalnum() for ch in token.text)]

        # unigrams
        for token in tokens:
            features[f"UNI_{token}"] += 1
        # bigrams
        for i in range(len(tokens) - 1):
            features[f"BI_{tokens[i]}_{tokens[i+1]}"] += 1

    return dict(features)

# build feature dictionary for a review under configuration M0, M1, or M2
def extract_features(review, model="M0", lexicon=None):
    if model not in {"M0", "M1", "M2"}:
        raise ValueError("Model must be M0, M1 or M2")

    sentences = preprocess(review)

    # base M0 bag-of-words features
    features = bow_features(sentences)

    # add M1 prior polarity lexicon density features
    if model in {"M1", "M2"}:
        if lexicon is None:
            raise ValueError("A sentiment lexicon is required")

        apply_lexicon(sentences, lexicon)
        features.update(lexicon_features(sentences))

    # add M2 compositional shifter (intensifier, negation, irrealis) features
    if model == "M2":
        apply_shifters(sentences)
        features.update(shifter_features(sentences))

    return features