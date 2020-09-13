from io import StringIO
from unittest import TestCase

from src.datasets.biocreative_dataset import BiocreativeDataset


class TestBiocreativeDataset(TestCase):

    def test___getitem__(self):
        """
        Test case if item is returned
        :return:
        """
        # Arrange
        input_raw = """P00001606T0076 Comparison with alkaline phosphatases and 5-nucleotidase
P00008171T0000 Pharmacologic aspects of neonatal hyperbilirubinemia."""
        input_raw_handle = StringIO(input_raw)

        input_annotation = """P00001606T0076|14 33|alkaline phosphatases
P00001606T0076|37 50|5-nucleotidase"""
        input_annotation_handle = StringIO(input_annotation)
        sut = BiocreativeDataset(input_raw_handle, input_annotation_handle)

        expected = ["Comparison with ", "alkaline phosphatases", " and ", "5-nucleotidase"], ["o", "s", "o", "s"]

        # Act
        actual = sut[0]

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test___getitem___no_entity(self):
        """
        Test case if item is returned with no entity
        :return:
        """
        # Arrange
        input_raw = """P00001606T0076 Comparison with alkaline phosphatases and 5-nucleotidase
P00008171T0000 Pharmacologic aspects of neonatal hyperbilirubinemia."""
        input_raw_handle = StringIO(input_raw)

        input_annotation = """P00001606T0076|14 33|alkaline phosphatases
P00001606T0076|37 50|5-nucleotidase"""
        input_annotation_handle = StringIO(input_annotation)
        sut = BiocreativeDataset(input_raw_handle, input_annotation_handle)

        expected = ["Pharmacologic aspects of neonatal hyperbilirubinemia."], ["o"]

        # Act
        actual = sut[1]

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test___len__(self):
        """
        Test case : Total number of samples in the dataset
        :return:
        """
        # Arrange
        input_raw = """P00001606T0076 Comparison with alkaline phosphatases and 5-nucleotidase
P00008171T0000 Pharmacologic aspects of neonatal hyperbilirubinemia."""
        input_raw_handle = StringIO(input_raw)

        input_annotation = """P00001606T0076|14 33|alkaline phosphatases"""
        input_annotation_handle = StringIO(input_annotation)
        sut = BiocreativeDataset(input_raw_handle, input_annotation_handle)


        expected = 2

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(expected, actual)
