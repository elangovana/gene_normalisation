import logging
import os

from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from collate import collate
from locator import Locator
from loss.ner_cross_entropy_loss import NerCrossEntropyLoss
from model.bert_model import BertModel
from preprocessor import Preprocessor
from trainer import Train


class Builder:

    def __init__(self, train_data, train_annotation_file, model_dir, dataset_factory_name, epochs=10, val_data=None,
                 val_annotation_file=None,
                 early_stopping_patience=5,
                 checkpoint_frequency=3, checkpoint_dir=None, learning_rate=0.0001, batch_size=8, max_seq_len=512,
                 fine_tune=False, num_workers=None, grad_accumulation_steps=4):

        self.grad_accumulation_steps = grad_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.model_dir = model_dir
        self.fine_tune = fine_tune
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_annotation_file = train_annotation_file
        self.train_data = train_data
        self._num_workers = num_workers or os.cpu_count() - 1
        self._max_seq_len = max_seq_len
        self._bert_model_name = "bert-base-cased"
        self._token_lower_case = False

        if self._num_workers <= 0:
            self._num_workers = 0

        self._trainer = None
        self._label_mapper = None
        self._train_dataloader = None
        self._train_dataset = None
        self._val_dataset = None
        self._network = None
        self._optimiser = None
        self._lossfunc = None
        self._bert_config = None
        self._tokenisor = None
        self._dataset_factory = Locator().get(dataset_factory_name)
        self.val_annotation_file = val_annotation_file
        self.val_data = val_data

    def set_bert_config(self, value):
        self._bert_config = value

    def set_tokensior(self, value):
        self._tokenisor = value

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_preprocessor(self):
        tokeniser = self._tokenisor
        if tokeniser is None:
            tokeniser = BertTokenizer.from_pretrained(self._bert_model_name, do_lower_case=self._token_lower_case,
                                                      do_basic_tokenize=False)

        preprocessor = Preprocessor(max_feature_len=self._max_seq_len, tokeniser=tokeniser,
                                    label_mapper=self.get_label_mapper())
        return preprocessor

    def get_label_mapper(self):
        if self._label_mapper is None:
            self._label_mapper = self._dataset_factory.get_label_mapper()
        return self._label_mapper

    def get_pos_label_index(self):
        return self.get_label_mapper().positive_label_index

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self._dataset_factory.get_dataset(self.train_data, self.train_annotation_file,
                                                                    preprocessors=self.get_preprocessor())

        return self._train_dataset

    def get_val_dataset(self):
        if self.val_data is None: return None

        if self._val_dataset is None:
            self._val_dataset = self._dataset_factory.get_dataset(self.val_data, self.val_annotation_file,
                                                                  preprocessors=self.get_preprocessor())

        return self._val_dataset

    def get_train_val_dataloader(self):
        if self._train_dataloader is None:
            # Split train into train and val
            train = self.get_train_dataset()
            val = self.get_val_dataset()

            if not val:
                # TODO stratify
                train, val = train_test_split(train, test_size=0.8, random_state=42)

            self._train_dataloader = DataLoader(dataset=train, num_workers=self._num_workers,
                                                batch_size=self.batch_size, shuffle=True, collate_fn=collate)

            self._val_dataloader = DataLoader(dataset=val, num_workers=self._num_workers,
                                              batch_size=self.batch_size, shuffle=False, collate_fn=collate)

        return self._train_dataloader, self._val_dataloader

    def get_network(self):
        # If network already loaded simply return
        if self._network is not None: return self._network

        if self._has_checkpoint():
            self._bert_model_name = self.checkpoint_dir

        self._network = BertModel(self._bert_model_name, self.get_label_mapper().num_classes,
                                  fine_tune=self.fine_tune, bert_config=self._bert_config)

        return self._network

    def get_loss_function(self):
        if self._lossfunc is None:
            k = int(self.batch_size)
            # self._lossfunc = TopKCrossEntropyLoss(k)
            self._lossfunc = NerCrossEntropyLoss()
        return self._lossfunc

    def get_optimiser(self):
        if self._optimiser is None:
            self._optimiser = Adam(params=self.get_network().parameters(), lr=self.learning_rate)
        return self._optimiser

    def get_trainer(self):
        if self._trainer is None:
            self._trainer = Train(model_dir=self.model_dir, label_mapper=self.get_label_mapper(), epochs=self.epochs,
                                  early_stopping_patience=self.early_stopping_patience,
                                  checkpoint_frequency=self.checkpoint_frequency,
                                  checkpoint_dir=self.checkpoint_dir,
                                  accumulation_steps=self.grad_accumulation_steps)

        return self._trainer

    def _has_checkpoint(self):
        if self.checkpoint_dir is None: return False

        return len(os.listdir(self.checkpoint_dir)) > 0
