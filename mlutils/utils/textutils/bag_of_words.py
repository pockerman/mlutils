from typing import Dict, List, Callable, Tuple, Any
from collections import Counter

class BagOfWords(object):
    def __init__(self):
        self.bow: Dict = {}

    def __len__(self):
        return len(self.bow)

    def add_tokens(self, tokens: List[str]) -> None:

        for token in tokens:
            if token in self.bow:
                self.bow[token] += 1
            else:
                self.bow[token] = 1

    def add_text(self, text: str, tokenizer: Callable) -> None:
        tokens = tokenizer(text)
        self.add_tokens(tokens=tokens)

    def most_common(self, n: int) -> List[Tuple[Any, int]]:
        return Counter(self.bow).most_common(n)