import unittest
import pytest
from mlutils.utils.textutils import compute_word_count_vector, compute_ngram_vector

@pytest.fixture()
def get_test_corpus():
    return ["Algorithmic bias has been cited in",
            "cases ranging from election outcomes to the spread of online hate speech."]
def test_word_count_vector(get_test_corpus):

        vector = compute_word_count_vector(corpus=get_test_corpus)
        assert len(vector)  == len(get_test_corpus), f"Number of vectors should {len(get_test_corpus)} but is {len(vector)}"

def test_compute_ngram_vector(get_test_corpus):
    vector = compute_ngram_vector(corpus=get_test_corpus, n_gram=(2,4))
    assert len(vector) == len(get_test_corpus), f"Number of vectors should {len(get_test_corpus)} but is {len(vector)}"


