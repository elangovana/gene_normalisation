class BiocreativeNerLabelMapper:

    def __init__(self):
        self._classes=['o', 's','c']
        self._classes_dict = { c:i for i, c in enumerate(self._classes)}
        self._indices_dict = { i:c for i, c in enumerate(self._classes)}

    @property
    def num_classes(self):
        return 3

    def label_to_index(self, label:str)->int:
        return self._classes_dict[label]

    def index_to_label(self, index:int)->str:
        return self._indices_dict[index]

    @property
    def positive_label(self):
        return 's'
