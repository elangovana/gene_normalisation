from typing import List

import torch


class Preprocessor:

    def __init__(self, max_feature_len, tokeniser, label_mapper):
        self.label_mapper = label_mapper
        self._other_symbol = self.label_mapper.other_label
        self._continution_symbols = self.label_mapper.continuation_symbol
        self._entity_symbols = self.label_mapper.entity_labels
        self.max_feature_len = max_feature_len
        self.tokeniser = tokeniser
        self._x = None

    @staticmethod
    def pad_token():
        return "[PAD]"

    @staticmethod
    def eos_token():
        return "<EOS>"

    @staticmethod
    def unk_token():
        return "[UNK]"

    def __call__(self, x: List[str], y: List[str] = None):
        self._x = x
        self._y = y

        self._tokenise() \
            ._sequence_pad() \
            ._token_to_index() \
            ._to_label_index() \
            ._to_tensor()

        return self._x, self._y

    def tokenise(self, x, y=None):
        """
        Converts text and y to continuation to tokens e.g.
         ["The protein", "KLK3 kinase"] would become ["The", "pro" "##tien", "KL" ,"#K3", "kinase"]
         ["o","s"] becomes ["o","o", "o", "s", "sc", "sc"]
         """
        new_x = []
        new_y = None if y is None else []

        for i, xi in enumerate(x):
            x_tokens = self.tokeniser.tokenize(xi)
            new_x.extend(x_tokens)

            if y is None: continue
            yi = y[i]
            if len(x_tokens) > 0:
                y_tokens = [yi] + [self._continution_symbols.get(yi, yi)] * (len(x_tokens) - 1)
                new_y.extend(y_tokens)

        return new_x, new_y

    def _tokenise(self):

        new_x, new_y = self.tokenise(self._x, self._y)

        self._x = new_x
        self._y = new_y
        return self

    def _token_to_index(self):
        """
        Converts a string of token to corresponding indices. e.g. ["The", "pro"] would return [2,3]
        :return: self
        """
        result = self.tokeniser.convert_tokens_to_ids(self._x)
        self._x = result
        return self

    def _sequence_pad(self):
        """
        Converts the tokens to fixed size and formats it according to bert
        :return: self
        """
        old_x = self._x
        old_y = self._y

        new_x, new_y = self.pad(old_x, old_y)

        self._x = new_x
        self._y = new_y
        return self

    def pad(self, x, y=None):
        tokens = x[:self.max_feature_len - 2]
        pad_tokens = [self.pad_token()] * (self.max_feature_len - 2 - len(tokens))
        new_x = ['[CLS]'] + tokens + pad_tokens + ['[SEP]']
        new_y = None
        if y is not None:
            new_y = [self.pad_token()] + self._y[:self.max_feature_len - 2] + [self.pad_token()] * len(pad_tokens) + [
                self.pad_token()]
        return new_x, new_y

    def _to_label_index(self):
        """
        Converts list of int to tensor
        :return: self
        """
        if self._y is None: return self

        new_y = []
        old_y = self._y

        for i in old_y:
            # If the preprocessor introduced pad character, then customise the index to -1
            index = -1 if i == self.pad_token() else self.label_mapper.label_to_index(i)
            new_y.append(index)

        self._y = new_y

        return self

    def _to_tensor(self):
        """
        Converts list of int to tensor
        :return: self
        """

        old_x = self._x
        old_y = self._y

        self._x = torch.tensor(old_x)
        self._y = torch.tensor(old_y) if old_y is not None else None
        return self
