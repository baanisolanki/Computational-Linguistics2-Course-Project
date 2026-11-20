to run:
python3 -m venv .venv
source .venv/bin/activate
pip install nltk scikit-learn numpy
python3 -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet', 'opinion_lexicon'])"
python3 test_components.py
python3 run_mid_evaluation.py


Dataset Curation & Linguistic Preprocessing (preprocess.py)
This is basically the first step of the entire pipeline. We preprocess the reviews by splitting them into sentences and words, performing Penn Treebank POS tagging, and lemmatizing them using WordNet (with POS guidance). We also keep track of clause boundaries within sentences using clause_id, since negation and other sentiment shifters should not affect words across different clauses. Additionally, third-person pronouns (PRP, PRP$) are identified and stored as part of the token information for further linguistic analysis.

Model M0 — Surface Bag-of-Words Baseline (features.py)
M0 is our basic baseline model. It uses unigram (UNI_<lemma>) and adjacent bigram (BI_<lemma1>_<lemma2>) frequency counts to represent each review. Basically, it looks at which words and word combinations occur and how frequently they appear, without taking their actual sentiment or linguistic context into account.

Model M1 — Prior Polarity Lexicon Semantics (lexicon_semantics.py)
M1 adds actual sentiment information to the features extracted in M0. We use the Hu & Liu Opinion Lexicon to assign prior polarity values to words (+1.0 for positive words and -1.0 for negative words). There is also a surface-word fallback for cases where words ending in -ing or -ed are lemmatized as verbs and might otherwise be missed during lexicon lookup.

Instead of simply counting positive and negative words, we calculate length-normalized document density features. This helps account for differences in review lengths and gives us a better idea of how much sentiment is present in a review.

Model M2 — Compositional Semantic Shifters (compositional_rules.py & config.py)
M2 builds on M1 by trying to capture some of the context in which opinion words are used. Just identifying a positive or negative word is not always enough, since its polarity can change depending on the words around it.

We implement three types of compositional shifters:
Intensifiers and downtoners: Words like "very" and "slightly" increase or decrease the polarity of the next opinion word within a 3-token window.
Negation: Words like "not", "never", and "hardly" shift the polarity of opinion words within the same clause and a 6-token window. We use an additive negation shift (NEGATION_SHIFT = 1.5) instead of simply flipping the polarity.
Irrealis modals: Words like "would", "could", and "might" dampen the polarity of opinion words that fall within their scope, using IRREALIS_FACTOR = 0.35.

The main idea here is to make the sentiment features a little more context-aware, rather than treating every opinion word independently.

Discourse Connective Taxonomy & Statistical/Rule-Based Classifiers (connectives.py, logistic_regression_model.py, naive_bayes_model.py, socal.py)
We also include an explicit discourse connective taxonomy based on RST-style rhetorical relations. The connectives are grouped into categories like Cause, Result, Contrast, Concession, Condition, Temporal, and Addition. This gives us a way to categorise discourse markers according to the kind of relationship they express.

For the classification part, we evaluate three different approaches:
L2-Regularized Logistic Regression: A statistical classifier trained on the extracted feature dictionaries.
Multinomial Naive Bayes: Another statistical classifier, which is evaluated using the same M0, M1, and M2 feature sets.
SO-CAL: A rule-based sentiment scorer that uses lexicon polarity, clause-bounded negation, and intensifier/downtoner adjustments to calculate the overall sentiment of a review.

Finally, we compare the different models across the train, validation, and test splits using metrics like accuracy, precision, recall, and F1-score. This lets us see whether adding linguistic information and compositional features actually improves sentiment classification, instead of just assuming that more features will give us better results.