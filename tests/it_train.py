import os
import pickle
import tempfile
from unittest import TestCase

from sklearn.model_selection import train_test_split

from builder import Builder
from datasets.biocreative_dataset import BiocreativeDataset
from datasets.biocreative_ner_label_mapper import BiocreativeNerLabelMapper
from serve import model_fn, input_fn, predict_fn


class ItTrain(TestCase):
    """
    This is a integration test , requires internet connectivity..
    """

    def test_run_train_predictions_match_inference(self):
        """
        This test makes sure train predictions match inference
        :return:
        """
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "data", "train.in")
        classes_file = os.path.join(os.path.dirname(__file__), "data", "train.eval")
        tempdir = tempfile.mkdtemp()
        batch = 3
        sequence_len = 512

        # Builder
        b = Builder(train_data=train_data_file, annotation_file=classes_file,
                    checkpoint_dir=None, epochs=1,
                    early_stopping_patience=1, batch_size=batch,
                    max_seq_len=sequence_len, model_dir=tempdir, num_workers=0)

        trainer = b.get_trainer()

        # Get data loaders
        train_dataloader, val_dataloader = b.get_train_val_dataloader()

        # Persist mapper so it case be used in inference
        label_mapper_pickle_file = os.path.join(tempdir, "label_mapper.pkl")
        with open(label_mapper_pickle_file, "wb") as f:
            pickle.dump(b.get_label_mapper(), f)

        # Persist tokensier
        preprocessor_pickle_file = os.path.join(tempdir, "preprocessor.pkl")
        with open(preprocessor_pickle_file, "wb") as f:
            pickle.dump(b.get_preprocessor(), f)

        # Run training
        scores = trainer.run_train(train_iter=train_dataloader,
                                   validation_iter=val_dataloader,
                                   model_network=b.get_network(),
                                   loss_function=b.get_loss_function(),
                                   optimizer=b.get_optimiser(), pos_label=b.get_pos_label_index())

        model = model_fn(tempdir)

        dataset = BiocreativeDataset(train_data_file, classes_file)
        train, val = train_test_split(dataset, test_size=0.8, random_state=42)

        val_pred = scores[2]
        for i in range(len(val)):
            input = input_fn("".join(val[i][0]), "text/csv")
            r = predict_fn(input, model)
            predicted_indices = [BiocreativeNerLabelMapper().label_to_index(v["entity"]) for v in r[0]]
            self.assertSequenceEqual(val_pred[i], predicted_indices)
