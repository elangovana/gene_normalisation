import os
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock

import transformers

from builder import Builder


class TestSitTrain(TestCase):

    def test_run_with_no_exception(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "data", "train.in")
        classes_file = os.path.join(os.path.dirname(__file__), "data", "train.eval")
        tempdir = tempfile.mkdtemp()
        batch = 3

        # Bert Config
        vocab_size = 20000
        sequence_len = 20
        num_classes = 3
        config = transformers.BertConfig(vocab_size=vocab_size, hidden_size=10, num_hidden_layers=1,
                                         num_attention_heads=1, num_labels=num_classes)

        # Mock tokenisor
        tokenisor = MagicMock()
        tokenisor.tokenize.side_effect = lambda x: x.split(" ")
        tokenisor.convert_tokens_to_ids = lambda x: [i for i, _ in enumerate(x)]

        # Builder
        b = Builder(train_data=train_data_file, annotation_file=classes_file,
                    checkpoint_dir=tempdir, epochs=2,
                    early_stopping_patience=2, batch_size=batch,
                    max_seq_len=sequence_len, model_dir=tempdir, tokeniser=tokenisor, bert_config=config)

        trainer = b.get_trainer()

        # Run training
        train_dataloader, val_dataloader = b.get_train_val_dataloader()


        # Act
        trainer.run_train(train_iter=train_dataloader,
                          validation_iter=val_dataloader,
                          model_network=b.get_network(),
                          loss_function=b.get_loss_function(),
                          optimizer=b.get_optimiser(), pos_label=b.get_pos_label_index())