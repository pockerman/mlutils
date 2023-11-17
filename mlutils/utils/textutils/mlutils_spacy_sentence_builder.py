from spacy import Language
from typing import List, Union, Iterator

from spacy.tokens import Span


class SpaCySentenceBuilder(object):

    def __init__(self, nlp: Language, expand: bool = True):
        self.nlp = nlp
        self.expand = expand

    def __call__(self, document: str) -> Union[List[str],Iterator[Span | Span]]:
        return self.run(document)

    def run(self, document: str) -> Union[List[str],Iterator[Span | Span]]:
        doc = self.nlp(document)

        if self.expand:
            return [str(sentence) for sentence in doc.sents]

        return doc.sents
