import re

from sklearn.feature_extraction.text import CountVectorizer

from ml.domain import normalize_domain  # noqa: F401  (re-export)

NGRAM_RANGE = (2, 5)
MIN_DF = 2
MAX_FEATURES = 200_000


def make_vectorizer():
    return CountVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        lowercase=False,
    )
