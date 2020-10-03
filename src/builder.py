import logging
import os

from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from collate import collate
from datasets.biocreative_dataset import BiocreativeDataset
from datasets.biocreative_ner_label_mapper import BiocreativeNerLabelMapper
from model.bert_model import BertModel
from preprocessor import Preprocessor
from trainer import Train
from top_k_cross_entropy_loss import TopKCrossEntropyLoss



class Builder:

    def __init__(self, train_data, annotation_file, model_dir, epochs=10, early_stopping_patience=5,
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
        self.annotation_file = annotation_file
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
        self._train_dataset=None
        self._network = None
        self._optimiser=None
        self._lossfunc=None
        self._bert_config = None
        self._tokenisor = None


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
            tokeniser = BertTokenizer.from_pretrained(self._bert_model_name, do_lower_case=self._token_lower_case)

        preprocessor = Preprocessor(max_feature_len=self._max_seq_len, tokeniser=tokeniser, label_mapper=self.get_label_mapper())
        return preprocessor

    def get_label_mapper(self):
        if self._label_mapper is None:
            self._label_mapper = BiocreativeNerLabelMapper()
        return self._label_mapper

    def get_pos_label_index(self):
        return self.get_label_mapper().positive_label_index

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = BiocreativeDataset(self.train_data, annotation_file_or_handle=self.annotation_file,
                                                     transformer=self.get_preprocessor())

        return self._train_dataset

    def get_train_val_dataloader(self):
        if self._train_dataloader is None:
            # Split train into train and val
            dataset = self.get_train_dataset()
            # TODO stratify
            train, val=  train_test_split(dataset, test_size=0.8, random_state=42)

            self._train_dataloader = DataLoader(dataset=train, num_workers=self._num_workers,
                                                batch_size=self.batch_size, shuffle=True, collate_fn=collate)

            self._val_dataloader = DataLoader(dataset=val, num_workers=self._num_workers,
                                              batch_size=self.batch_size, shuffle=False,collate_fn=collate)

        return self._train_dataloader, self._val_dataloader

    def get_network(self):
        # If network already loaded simply return
        if self._network is not None: return self._network

        # If checkpoint file is available, load from checkpoint
        model_weights = self.get_trainer().try_load_model_from_checkpoint()

        self._network = BertModel(self._bert_model_name, self.get_label_mapper().num_classes,
                                  fine_tune=self.fine_tune, bert_config= self._bert_config)

        #  Load checkpoint when checkpoint is available
        if model_weights   is not  None:
            self._network.load_state_dict(model_weights)

        return self._network

    def get_loss_function(self):
        if self._lossfunc is None:
            k = int(self.batch_size/2)
            self._lossfunc = TopKCrossEntropyLoss(k)
            #self._lossfunc  = nn.CrossEntropyLoss()
        return self._lossfunc

    def get_optimiser(self):
        if self._optimiser is None:
            self._optimiser = Adam(params=self.get_network().parameters(), lr=self.learning_rate)
        return self._optimiser

    def get_trainer(self):
        if self._trainer is None:
            self._trainer = Train(model_dir=self.model_dir,label_mapper=self.get_label_mapper(), epochs=self.epochs,
                                  early_stopping_patience=self.early_stopping_patience,
                                  checkpoint_frequency=self.checkpoint_frequency,
                                  checkpoint_dir=self.checkpoint_dir,
                                  accumulation_steps=self.grad_accumulation_steps)

        return self._trainer
