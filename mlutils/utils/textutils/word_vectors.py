from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer

def compute_word_count_vector(corpus: List[str]) -> List:
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(corpus)
    return vector.toarray()

def compute_ngram_vector(corpus: List[str], n_gram: Tuple[int, int]) -> List:
    vectorizer = CountVectorizer(ngram_range=n_gram)
    vector = vectorizer.fit_transform(corpus)
    return vector.toarray()


