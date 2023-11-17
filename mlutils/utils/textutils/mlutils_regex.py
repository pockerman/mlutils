import re
from typing import List, Union

class ApplyRegex(object):
    def __init__(self, patterns: List[str]):
        self.patterns = patterns

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:

        if isinstance(text, str):
            return self.apply_regex(text=text)

        result = []
        for i, t in enumerate(text):
            result.append(self.apply_regex(text=t))
        return result

    def apply_regex(self, text: str) -> str:

        for pattern in self.patterns:
            matches = re.finditer(pattern, text)

            if matches:
                for match in matches:
                    text = text.replace(match.group(0), "")
        return text