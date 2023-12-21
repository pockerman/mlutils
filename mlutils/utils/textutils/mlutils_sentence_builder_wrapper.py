from abc import ABC, abstractmethod
from typing import List, Union, Iterator
import re
from spacy import Language
from spacy.tokens import Span


class SentenceBuilderWrapperBase(ABC):
    def __init__(self):
        self.sentences: List = []

    def __call__(self, document: str) -> Union[List[str],Iterator[Span | Span]]:
        return self.run(document)
    @abstractmethod
    def run(self, document: str) -> Union[List[str],Iterator[Span | Span]]:
        pass


class RuleBasedSentenceBuilderWrapper(SentenceBuilderWrapperBase):
    def __init__(self, re_rule: str = r'[!.?]+[\s$]+'):
        super(RuleBasedSentenceBuilderWrapper, self).__init__()
        self.re_rule = re_rule

    def run(self, document: str) -> Union[List[str],Iterator[Span | Span]]:

        self.sentences =  re.split(self.re_rule, document)
        return self.sentences

class SpaCySentenceBuilder(SentenceBuilderWrapperBase):
    """class SpaCySentenceBuilder. Uses spaCy
    to find sentences in a given piece of text

    """
    def __init__(self, nlp: Language, expand: bool = True):
        super(SentenceBuilderWrapperBase, self).__init__()
        self.nlp = nlp
        self.expand = expand

    def run(self, document: str) -> Union[List[str],Iterator[Span | Span]]:
        doc = self.nlp(document)

        if self.expand:
            self.sentences = [str(sentence) for sentence in doc.sents]
        else:
            self.sentences = doc.sents

        return self.sentences
