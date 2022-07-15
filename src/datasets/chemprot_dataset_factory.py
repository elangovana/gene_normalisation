from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.biocreative_dataset import BiocreativeDataset
from datasets.chemprot_dataset import ChemprotDataset
from datasets.chemprot_ner_label_mapper import ChemprotNerLabelMapper


class ChemprotDatasetFactory(BaseDatasetFactory):

    def get_dataset(self, data, annotation_data=None, preprocessors=None, **kwargs):
        return ChemprotDataset(data, annotation_file_or_handle=annotation_data,
                           transformer=preprocessors)

    def get_label_mapper(self):
        return ChemprotNerLabelMapper()

