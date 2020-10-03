import logging

import torch
from torch import nn


class NerCrossEntropyLoss(nn.Module):
    """
    Computes the Cross entropy loss for the top k for ner to take into account pad
    """

    def __init__(self, pad_index_label=-1):
        super().__init__()
        self.pad_index_label = pad_index_label

        self._loss_func = nn.CrossEntropyLoss(reduction='none')

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def forward(self, predicted, target):
        # remove all entries where the target is -1 (pseudo index), indicating predictions for PAD character that we dont really care about
        mask = (target != self.pad_index_label).long().to(device=predicted.device)
        p_mask = torch.unsqueeze(mask, dim=1)

        # zero out the values for pad index in target and predicted, so that they don't count
        predicted_zerod = predicted * p_mask
        target_zerod = target * mask

        loss_per_item = (self._loss_func(predicted_zerod, target_zerod) * mask)

        # Only compute loss where mask is non zero
        non_zero_average_loss_per_item = loss_per_item.sum(dim=1) / mask.sum(dim=1)

        loss = torch.mean(non_zero_average_loss_per_item)

        return loss
