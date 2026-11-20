# hyperparameter settings for sentiment lexicon and compositional shifters
SEED=42

# improvement 1: Changed NEGATION_SHIFT from 4.0 to 1.5 to match the -1.0 to +1.0 lexicon scale
NEGATION_SHIFT=1.5

# multipliers for degree intensifiers (>1.0) and downtoners (<1.0)
INTENSIFIERS={"very":1.5,"extremely":1.7,"utterly":1.8, "really":1.3,"slightly":0.5, "somewhat":0.7, "barely":0.5,"quite":1.2}

# modal and hypothetical markers that dampen sentiment (irrealis mood)
IRREALIS={ "would", "could","should", "might","may", "if","hope","expect"}

# dampening factor applied when an opinion word falls under irrealis scope
IRREALIS_FACTOR=0.35