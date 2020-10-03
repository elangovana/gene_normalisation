from torch import nn
from transformers import BertForTokenClassification


class BertModel(nn.Module):

    def __init__(self, model_name_or_dir, num_classes, fine_tune=True, state_dict=None, config=None):
        super().__init__()
        if config is None:
            self.model = BertForTokenClassification.from_pretrained(model_name_or_dir, num_labels=num_classes, state_dict=state_dict)
        else:
            self.model = BertForTokenClassification(config)

        # Fine tune, freeze all other weights except classifier
        if fine_tune:
            self._freeze_base_weights()

    def _freeze_base_weights(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, *input):
        return self.model(*input)