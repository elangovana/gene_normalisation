from torch import nn
from transformers import BertForTokenClassification


class BertModel(nn.Module):

    def __init__(self, pretrained_model_name_or_dir=None, pretrained_num_classes=None, fine_tune=False,
                 bert_config=None):
        """
        Buils a bert model for token classification
        :param pretrained_model_name_or_dir: Specify the pretrained_model_name_or_dir to load from to start from a pretrained model
        :param pretrained_num_classes: The number of classes for the pretrained model
        :param fine_tune: If fine tune is true, only the classification layer weights are tuned.
        :param bert_config: If this is not none, this config is used to create a BERT model from scratch using the configuration
        """
        super().__init__()
        if bert_config is None:
            self.model = BertForTokenClassification.from_pretrained(pretrained_model_name_or_dir, num_labels=pretrained_num_classes)
        else:
            self.model = BertForTokenClassification(bert_config)

        # Fine tune, freeze all other weights except classifier
        if fine_tune:
            self._freeze_base_weights()

    def _freeze_base_weights(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, *input):
        return self.model(*input)

    def save(self, path):
        self.model.save_pretrained(save_directory=path)
