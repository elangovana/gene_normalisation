from unittest import TestCase

import torch
from torch.nn import CrossEntropyLoss

from top_k_cross_entropy_loss import TopKCrossEntropyLoss


class TestTopKCrossEntropyLoss(TestCase):


    def test_forward(self):
        # Arrange
        k = 2
        predicted = torch.tensor([[[0.5, .5], [.7, 0.3], [0.0, 1.0]]
                                     , [[0.2, .8], [.8, 0.2], [0.0, 1.0]]
                                     , [[0.5, .5], [1.0, 0.0], [0.0, 1.0]]
                                  ])
        target = torch.tensor([[0, 1, 0]
                                  , [0, 1, 0]
                                  , [0, 1, 0]
                               ])
        indices_high_loss = torch.tensor([1, 2])

        expected_loss = CrossEntropyLoss()(predicted[indices_high_loss, :].permute(0, 2, 1),
                                           target[indices_high_loss, :])

        sut = TopKCrossEntropyLoss(k)

        # Act
        actual = sut.forward(predicted.permute(0,2,1), target)

        # Assert
        self.assertEqual(round(expected_loss.item(), 2), round(actual.item(), 2))

