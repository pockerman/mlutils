"""module spacy_tokenizer. Simple class that
wraps spaCy tokens

"""
import re
from typing import List, Union
from spacy import Language
from spacy.tokens import Token

class TextSplitTokenizerWrapper(object):
    """class TextSplitTokenizerWrapper. Simply splits
    the given text on the provided mark. The default splitting
    mark is the single space.
    """
    def __init__(self, mark: str=" "):
        self.mark = mark

    def __call__(self, text: str) -> List[str]:
        return text.split(self.mark)

class SpaCyTokenizerWrapper(object):
    def __init__(self, nlp: Language, out_as_strings: bool = True):
        self.nlp = nlp
        self.out_as_strings = out_as_strings

    def __call__(self, document: str) -> Union[List[str], List[Token]]:

        tokens = self.nlp(document)

        if self.out_as_strings:
            tokens = [token.text for token in tokens]
        return tokens

class RuleBasedTokenizerWrapper(object):
    def __init__(self, re_pattern: str):
        self.re_pattern = re_pattern

    def __call__(self, text: str) -> List[str]:
        return list(re.findall(self.re_pattern, text))
