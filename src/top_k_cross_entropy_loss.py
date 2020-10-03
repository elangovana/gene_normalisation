import logging

import torch
from torch import nn


class TopKCrossEntropyLoss(nn.Module):
    """
    Computes the Cross entropy loss for the top k most difficult samples
    """

    def __init__(self, k, pad_index_label=-1):
        super().__init__()
        self.pad_index_label = pad_index_label
        self.k = k
        self._loss_func = nn.CrossEntropyLoss(reduction='none')

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def forward(self, predicted, target):
        # remove all entries where the target is -1 (pseudo index), indicating predictions for PAD character that we dont really care about
        mask = (target != self.pad_index_label).long().to(device=predicted.device)

        p_mask =  torch.unsqueeze(mask, dim=1)

        # zero out the values for pad index in target and predicted, so that they don't count
        predicted_zerod = predicted * p_mask
        target_zerod = target * mask

        # make sure k is within the length of the target shape
        k = min(self.k, target_zerod.shape[0])

        loss_per_item = self._loss_func(predicted_zerod, target_zerod)

        # Obtain only the top k hard samples
        top_k_loss = torch.mean(torch.topk(loss_per_item, k=k)[0])
        self._logger.debug("Total loss {} vs topk loss {}".format(torch.mean(loss_per_item), top_k_loss))

        return top_k_loss
