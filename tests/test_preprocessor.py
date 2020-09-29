from unittest import TestCase
from unittest.mock import MagicMock

from src.preprocessor import Preprocessor


class TestPreprocessor(TestCase):
    def test__tokenise(self):
        # Arrange
        x = ["Comparison with", "alkaline phosphatases", "and", "5-nucleotidase"]
        y = ["o", "s", "o", "s"]

        expected_x = ["Comparison", "with", "alkaline", "phosphatases", "and", "5-nucleotidase"]
        expected_y = ["o", "o", "s", "sc", "o", "s"]

        tokensier = MagicMock()
        tokensier.tokenize.side_effect = lambda x: x.split(" ")

        sut = Preprocessor(max_feature_len=5, tokeniser=tokensier, entity_symbols=["s"],
                           continution_symbols={"s": "sc"})
        sut._x = x
        sut._y = y

        # Act
        sut._tokenise()

        # Assert
        self.assertSequenceEqual(expected_x, sut._x)
        self.assertSequenceEqual(expected_y, sut._y)
