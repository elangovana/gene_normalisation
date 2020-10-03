from unittest import TestCase

import torch
from torch.nn import CrossEntropyLoss

from ner_cross_entropy_loss import NerCrossEntropyLoss



class TestNerCrossEntropyLoss(TestCase):


    def test_forward(self):
        # Arrange
        k = 2
        pad_index = -1

        predicted = torch.tensor([
                                [[0.5, .5], [0.5, .6], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
                                  ,[[0.5, .5], [0.5, .6], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
                                  ])
        target = torch.tensor([
                                [pad_index, 0, 1, 0, pad_index]
                              , [pad_index, 0, 1, 0, pad_index]
                               ])

        non_pad_indices= torch.tensor([1, 2, 3])

        expected_loss = CrossEntropyLoss()(predicted[:,non_pad_indices].permute(0,2,1), target[:,non_pad_indices])

        sut = NerCrossEntropyLoss(pad_index_label=pad_index)

        # Act
        actual = sut.forward(predicted.permute(0,2,1), target)

        # Assert
        self.assertEqual(round(expected_loss.item(), 2), round(actual.item(), 2))