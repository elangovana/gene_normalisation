from typing import List, Dict

from datasets.base_label_mapper import BaseLabelMapper


class BiocreativeNerLabelMapper(BaseLabelMapper):

    def __init__(self):
        self._classes = ['O', 'B-GENE', 'I-GENE']
        self._classes_dict = {c: i for i, c in enumerate(self._classes)}
        self._indices_dict = {i: c for i, c in enumerate(self._classes)}

    @property
    def num_classes(self):
        return len(self._classes)

    def label_to_index(self, label: str) -> int:
        return self._classes_dict[label]

    def index_to_label(self, index: int) -> str:
        return self._indices_dict[index]

    @property
    def entity_labels(self) -> List[str]:
        return ["B-GENE"]

    @property
    def other_label(self) -> str:
        return "O"

    @property
    def continuation_symbol(self) -> Dict[str, str]:
        return {"B-GENE": "I-GENE"}

    @property
    def positive_label(self):
        return self.gene_label

    @property
    def positive_label_index(self):
        return self._classes_dict[self.positive_label]

    @property
    def gene_label(self):
        return 'B-GENE'
