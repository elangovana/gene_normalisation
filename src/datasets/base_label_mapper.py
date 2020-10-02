from typing import Dict, List


class BaseLabelMapper:

    @property
    def num_classes(self):
        raise NotImplementedError

    def label_to_index(self, label: str) -> int:
        raise NotImplementedError

    def index_to_label(self, index: int) -> str:
        raise NotImplementedError

    @property
    def entity_labels(self) -> List[str]:
        raise NotImplementedError

    @property
    def other_label(self) -> str:
        raise NotImplementedError

    @property
    def continuation_symbol(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def positive_label(self):
        raise NotImplementedError

    @property
    def positive_label_index(self):
        raise NotImplementedError