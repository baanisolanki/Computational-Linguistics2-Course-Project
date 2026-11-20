# lexicon semantics module (model M1): hu & liu opinion lexicon lookup and length-normalized features
from nltk.corpus import opinion_lexicon
from preprocess import flatten, ensure_nltk

# load positive (+1.0) and negative (-1.0) words from nltk opinion lexicon
def build_lexicon():
    # improvement 2a: Ensure NLTK data is ready before loading opinion_lexicon
    ensure_nltk()
    lexicon = {}

    for word in opinion_lexicon.positive():
        lexicon[word] = 1.0

    for word in opinion_lexicon.negative():
        lexicon[word] = -1.0

    return lexicon


# assign prior polarity from lexicon to each token in the review
def apply_lexicon(sentences, lexicon):
    for token in flatten(sentences):
        # improvement 2b: Check lowercase word first so "-ing"/"-ed" adjectives lemmatized as verbs are not missed
        polarity = lexicon.get(token.text.lower(), lexicon.get(token.lemma, 0.0))
        token.prior_polarity = polarity
        token.shifted_polarity = polarity

    return sentences


# compute length-normalized lexicon density features for model M1
def lexicon_features(sentences):
    tokens = flatten(sentences)

    # count alphanumeric words to normalize features by document length
    total_words = max(1,sum(1 for t in tokens if any(c.isalnum() for c in t.text)))

    positive = [t.prior_polarity for t in tokens if t.prior_polarity > 0]
    negative = [t.prior_polarity for t in tokens if t.prior_polarity < 0]

    return {
        "lex_pos_density": sum(positive) / total_words,
        "lex_neg_density": abs(sum(negative)) / total_words,
        "lex_pos_count_density": len(positive) / total_words,
        "lex_neg_count_density": len(negative) / total_words,
        "lex_total_density": (len(positive) + len(negative)) / total_words,
        "lex_polarity_density": sum(positive + negative) / total_words,
        # improvement 4a: Non-negative net polarity features so Multinomial Naive Bayes does not drop net polarity
        "lex_pos_excess_density": max(0.0, sum(positive + negative)) / total_words,
        "lex_neg_excess_density": max(0.0, -sum(positive + negative)) / total_words
    }