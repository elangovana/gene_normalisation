from io import StringIO
from unittest import TestCase

from datasets.biocreative_dataset import BiocreativeDataset
from datasets.chemprot_dataset import ChemprotDataset


class TestChemprotDataset(TestCase):

    def test__getitem__gene(self):
        """
        Test case for gene annotation
        :return:
        """
        # Arrange
        input_raw = """10471277	Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.	Salmeterol is a
23150485	Induction of multidrug resistance transporter ABCG2 by prolactin in human breast cancer cells.	The multidrug transporter, """
        input_raw_handle = StringIO(input_raw)

        input_annotation = """23150485	T42	GENE-Y	46	51	ABCG2
10471277	T40	GENE-Y	43	69	beta 2-adrenergic receptor"""
        input_annotation_handle = StringIO(input_annotation)
        sut = ChemprotDataset(input_raw_handle, input_annotation_handle)

        expected = ["Probing the salmeterol binding site on the ", "beta 2-adrenergic receptor", " using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol. Salmeterol is a"], ["O", "B-GENE", "O"]

        # Act
        actual = sut[0]

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test__getitem__chem(self):
            """
            Test case for chem annotation
            :return:
            """
            # Arrange
            input_raw = """10471277	Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.	Salmeterol is a
23150485	Induction of multidrug resistance transporter ABCG2 by prolactin in human breast cancer cells.	The multidrug transporter, """
            input_raw_handle = StringIO(input_raw)

            input_annotation = """23150485	T42	CHEMICAL	46	51	ABCG2
10471277	T40	GENE-Y	43	69	beta 2-adrenergic receptor"""
            input_annotation_handle = StringIO(input_annotation)
            sut = ChemprotDataset(input_raw_handle, input_annotation_handle)

            expected = ["Induction of multidrug resistance transporter ", "ABCG2", " by prolactin in human breast cancer cells. The multidrug transporter, "], ["O", "B-CHEMICAL", "O"]

            # Act
            actual = sut[1]

            # Assert
            self.assertSequenceEqual(expected, actual)

