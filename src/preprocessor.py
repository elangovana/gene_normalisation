from typing import List

import torch


class Preprocessor:

    def __init__(self, max_feature_len, tokeniser, entity_symbols, continution_symbols):
        self._continution_symbols = continution_symbols
        self._entity_symbols = entity_symbols
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
            ._to_tensor()

        return self._x, self._y

    def _tokenise(self):
        """
        Converts text and y to continuation to tokens e.g.
         ["The protein", "KLK3 kinase"] would become ["The", "pro" "##tien", "KL" ,"#K3", "kinase"]
         ["o","s"] becomes ["o","o", "o", "s", "sc", "sc"]
        """
        x = []
        y = []
        for xi, yi in zip(self._x, self._y):
            x_tokens = self.tokeniser.tokenize(xi)
            y_tokens = [yi] + [self._continution_symbols.get(yi, yi)] * (len(x_tokens) - 1)
            x.extend(x_tokens)
            y.extend(y_tokens)
        self._x = x
        self._y = y
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
        tokens = self._x[:self.max_feature_len - 2]
        pad_tokens = [self.pad_token()] * (self.max_feature_len - 2 - len(tokens))
        result = ['[CLS]'] + tokens + pad_tokens + ['[SEP]']

        self._x = result
        return self

    def _to_tensor(self):
        """
        Converts list of int to tensor
        :return: self
        """

        self._x = torch.tensor(self._x)
        return self
