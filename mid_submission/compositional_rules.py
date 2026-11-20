# compositional semantic shifters module (model M2): intensifiers, clause-bounded negation, and irrealis modals
from config import (INTENSIFIERS,IRREALIS,IRREALIS_FACTOR,NEGATION_SHIFT)

# negation cues that shift polarity of opinion words within the same clause
NEGATORS = {"not","n't","never","no","hardly"}

# apply intensifier scaling, negation shifting, and irrealis modal dampening within clause boundaries
def apply_shifters(sentences,intensifiers=None,negation_shift=NEGATION_SHIFT):
    if intensifiers is None:
        intensifiers=INTENSIFIERS

    for sentence in sentences:
        tokens= sentence.tokens

        # reset polarity before applying rules
        for token in tokens:
            token.shifted_polarity = token.prior_polarity

        # improvement 1b: applying intensifiers before negation, and break after negating the first target word
        # pass 1: scale opinion words modified by intensifiers or downtoners (3-token window in same clause)
        for i, token in enumerate(tokens):
            word=token.lemma.lower()
            if word in intensifiers:
                for next_token in tokens[i+1:i+4]:
                    if next_token.clause_id != token.clause_id:
                        break
                    if next_token.prior_polarity != 0:
                        next_token.shifted_polarity *= (intensifiers[word])
                        break

        # pass 2: apply clause-bounded negation shift and irrealis modal dampening (6-token window)
        for i, token in enumerate(tokens):
            word = token.lemma.lower()
            if word in NEGATORS:
                for next_token in tokens[i+1:i+7]:      
                    if next_token.clause_id != token.clause_id:
                        break
                    if next_token.text in {",", ";", ".", "!"}:
                        break
                    if next_token.prior_polarity != 0:
                        if next_token.shifted_polarity > 0:
                            next_token.shifted_polarity = (next_token.shifted_polarity- negation_shift)
                        else:
                            next_token.shifted_polarity=(next_token.shifted_polarity+ negation_shift)
                        break

            # dampen polarity if opinion word falls under irrealis/modal scope
            if word in IRREALIS:
                for next_token in tokens[i+1:i+7]:
                    if next_token.clause_id != token.clause_id:
                        break
                    if next_token.prior_polarity != 0:
                        next_token.shifted_polarity *= IRREALIS_FACTOR

    return sentences


# compute length-normalized shifted polarity features for model M2
def shifter_features(sentences):
    tokens =[t for sentence in sentences for t in sentence.tokens]
    total_words=max(1,sum(1 for t in tokens if any(c.isalnum() for c in t.text)))
    values = [t.shifted_polarity for t in tokens if t.prior_polarity!= 0]

    return {
        "shift_pos_density": sum(max(0, x) for x in values) / total_words,
        "shift_neg_density": sum(max(0, -x) for x in values) / total_words,
        "shift_density": sum(values) / total_words,
        "shift_abs_density": sum(abs(x) for x in values) / total_words,
        # improvement 4b: Non-negative net shifted polarity features for Multinomial Naive Bayes
        "shift_pos_excess_density": max(0.0, sum(values)) / total_words,
        "shift_neg_excess_density": max(0.0, -sum(values)) / total_words
    }
