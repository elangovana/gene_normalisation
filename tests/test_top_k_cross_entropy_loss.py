from unittest import TestCase

import torch
from torch.nn import CrossEntropyLoss

from top_k_cross_entropy_loss import TopKCrossEntropyLoss


class TestTopKCrossEntropyLoss(TestCase):
    def test_forward_one_item(self):
        # Arrange
        k = 1
        predicted = torch.tensor([[0.1, 0.9]])
        target = torch.tensor([0])
        expected_loss = CrossEntropyLoss()(predicted, target)

        sut = TopKCrossEntropyLoss(k)

        # Act
        actual = sut.forward(predicted, target)

        # Assert
        self.assertEqual(round(expected_loss.item(), 2), round(actual.item(), 2))

    def test_forward(self):
        # Arrange
        k = 2
        predicted = torch.tensor([[[0.5, .5], [1.0, 0.0], [0.0, 1.0]]])
        target = torch.tensor([[0, 1, 0]])
        indices_high_loss = torch.tensor([1, 2])

        expected_loss = CrossEntropyLoss()(predicted[:, indices_high_loss], target[:, indices_high_loss])

        sut = TopKCrossEntropyLoss(k)

        # Act
        actual = sut.forward(predicted.permute(0,2,1), target)

        # Assert
        self.assertEqual(round(expected_loss.item(), 2), round(actual.item(), 2))

    def test_forward_pseudo_index_batch_sequence(self):
        # Arrange
        k = 2
        pseudo_index = -1

        predicted = torch.tensor([[[0.5, .5], [0.5, .5], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
        target = torch.tensor([[pseudo_index, 0, 1, 0, pseudo_index]])

        indices_high_loss = torch.tensor([2, 3])

        expected_loss = CrossEntropyLoss()(predicted[:,indices_high_loss], target[:,indices_high_loss])

        sut = TopKCrossEntropyLoss(k, pad_index_label=pseudo_index)

        # Act
        actual = sut.forward(predicted.permute(0,2,1), target)

        # Assert
        self.assertEqual(round(expected_loss.item(), 2), round(actual.item(), 2))