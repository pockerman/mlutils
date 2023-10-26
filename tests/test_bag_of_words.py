import unittest
from mlutils.utils.textutils import BagOfWords


class TestImageTransformers(unittest.TestCase):

    def test_as_pandas_df(self):

        sentence = ("Algorithmic bias has been cited in cases "
                    "ranging from election outcomes to the spread of online hate speech.")

        tokens = sentence.split(" ")
        bow = BagOfWords()
        bow.add_tokens(tokens)

        self.assertEqual(len(tokens), len(bow))

        # create the dataframe
        df = bow.as_pandas_df(fill_na=True)
        self.assertEqual(len(tokens), len(df.columns))
