class BaseDatasetFactory:
    def get_dataset(self, data, preprocessors=None, **kwargs):
        raise NotImplementedError

    def get_label_mapper(self):
        raise NotImplementedError

    # def get_scorers(self):
    #     raise NotImplementedError