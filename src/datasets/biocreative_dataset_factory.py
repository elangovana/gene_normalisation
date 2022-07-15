from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.biocreative_dataset import BiocreativeDataset
from datasets.biocreative_ner_label_mapper import BiocreativeNerLabelMapper


class BiocreativeDatasetFactory(BaseDatasetFactory):

    def get_dataset(self, data, annotation_data=None, preprocessors=None, **kwargs):
        return BiocreativeDataset(data, annotation_file_or_handle=annotation_data,
                           transformer=preprocessors)


    def get_label_mapper(self):
        return BiocreativeNerLabelMapper()