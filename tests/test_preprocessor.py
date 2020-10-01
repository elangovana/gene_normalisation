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

        label_mapper=MagicMock()
        label_mapper.entity_labels=["s"]
        label_mapper.continuation_symbol={"s": "sc"}
        label_mapper.other_label="o"

        sut = Preprocessor(max_feature_len=5, tokeniser=tokensier, label_mapper=label_mapper)
        sut._x = x
        sut._y = y

        # Act
        sut._tokenise()

        # Assert
        self.assertSequenceEqual(expected_x, sut._x)
        self.assertSequenceEqual(expected_y, sut._y)

    def test__tokenise_label_len_matches_tokens(self):
        # Arrange
        x = ["Comparison with", "alkaline phosphatases", "and"," "," ", "5-nucleotidase"]
        y = ["o", "s", "o","o","o", "s"]

        # expected_x = ["Comparison", "with", "alkaline", "phosphatases", "and", "5-nucleotidase"]
        # expected_y = ["o", "o", "s", "sc", "o", "s"]

        tokensier = MagicMock()
        tokensier.tokenize.side_effect = lambda z: list(filter(lambda y:y!="",z.split(" ")))

        label_mapper=MagicMock()
        label_mapper.entity_labels=["s"]
        label_mapper.continuation_symbol={"s": "sc"}
        label_mapper.other_label="o"

        sut = Preprocessor(max_feature_len=5, tokeniser=tokensier, label_mapper=label_mapper)
        sut._x = x
        sut._y = y

        # Act
        sut._tokenise()
        print(sut._x)
        print(sut._y)

        # Assert
        self.assertEqual(len(sut._x), len(sut._y))

    def test__sequence_pad(self):
        """Test case shorter sequence with padding"""
        # Arrange
        x = ["Comparison", "with", "alkaline", "phosphatases", "and", "5-nucleotidase"]
        y = ["o", "o", "s", "sc", "o", "s"]

        expected_x = ['[CLS]', "Comparison", "with", "alkaline", "phosphatases", "and", "5-nucleotidase", "[PAD]",
                      "[PAD]",
                      "[SEP]"]
        expected_y = ["o", "o", "o", "s", "sc", "o", "s", "o", "o", "o"]

        tokensier = MagicMock()
        tokensier.tokenize.side_effect = lambda x: x.split(" ")

        label_mapper = MagicMock()
        label_mapper.entity_labels = ["s"]
        label_mapper.continuation_symbol = {"s": "sc"}
        label_mapper.other_label = "o"

        sut = Preprocessor(max_feature_len=10, tokeniser=tokensier, label_mapper=label_mapper)
        sut._x = x
        sut._y = y

        # Act
        sut._sequence_pad()

        # Assert
        self.assertSequenceEqual(expected_x, sut._x)
        self.assertSequenceEqual(expected_y, sut._y)

    def test__sequence_too_long(self):
        """Test case longer sequence """
        # Arrange
        x = ["Comparison", "with", "alkaline", "phosphatases", "and", "5-nucleotidase"]
        y = ["o", "o", "s", "sc", "o", "s"]

        expected_x = ['[CLS]', "Comparison", "with", "alkaline", "[SEP]"]
        expected_y = ["o", "o", "o", "s", "o"]

        tokensier = MagicMock()
        tokensier.tokenize.side_effect = lambda x: x.split(" ")

        label_mapper = MagicMock()
        label_mapper.entity_labels = ["s"]
        label_mapper.continuation_symbol = {"s": "sc"}
        label_mapper.other_label = "o"

        sut = Preprocessor(max_feature_len=5, tokeniser=tokensier, label_mapper=label_mapper)
        sut._x = x
        sut._y = y

        # Act
        sut._sequence_pad()

        # Assert
        self.assertSequenceEqual(expected_x, sut._x)
        self.assertSequenceEqual(expected_y, sut._y)
