from torch.utils.data import Dataset

from datasets.chemprot_ner_label_mapper import ChemprotNerLabelMapper


class ChemprotDataset(Dataset):
    """
        Chemprot dataset
    """

    def __init__(self, train_file_or_handle, annotation_file_or_handle=None, transformer=None):
        # Load raw train
        self.transformer = transformer
        self._text_lines = self._parse(self._readlines(train_file_or_handle))
        self.label_mapper = ChemprotNerLabelMapper()

        # Load annotations
        self._annotation = {}
        if annotation_file_or_handle is not None:
            self._annotation = self._parse_annotation(self._readlines(annotation_file_or_handle))

    def _readlines(self, file_or_handle):
        if isinstance(file_or_handle, str):
            with open(file_or_handle, "r") as f:
                lines = f.readlines()
        else:
            lines = file_or_handle.readlines()

        return lines

    def __len__(self):
        return len(self._text_lines)

    def __getitem__(self, item):
        id, text = self._text_lines[item]["id"], self._text_lines[item]["text"]
        annotations = self._annotation.get(id, None)

        token_text, token_labels = self._tokenise(text, annotations)

        if self.transformer is not None:
            token_text, token_labels = self.transformer(token_text, token_labels)

        return token_text, token_labels

    def _parse(self, raw_lines):
        result = []
        for r in raw_lines:
            id, title, text = r.split("\t")[0],r.split("\t")[1], r.split("\t")[2]

            result.append({
                "id": id,
                "text":  title+ " "+text.rstrip("\n")
            })
        return result

    def _parse_annotation(self, annotation_lines):
        result = {}
        for l in annotation_lines:
            id, _, type, start, end, text = l.split("\t")
            text = text.rstrip("\n")

            if id not in result:
                result[id] = []

            result[id].append({
                "start": int(start)
                , "end": int(end)
                , "text": text
                , "type": type if type == 'CHEMICAL' else type[:4]
            })
        return result

    def _tokenise(self, line, annotation=None):
        if annotation is None:
            return [line], [self.label_mapper.other_label]

        tokens = []
        tokens_labels = []

        sorted_annotations = sorted(annotation, key=lambda item: item["start"])

        i = 0
        for item in sorted_annotations:
            # Find true position within approximate locations specified
            token_span = line[item["start"]:]
            span_index = token_span.find(item["text"])
            length = len(item["text"])
            anno_type = item["type"]

            assert span_index > -1, "Could not find {} in span {} in line {}, {}".format(item["text"], token_span, line,
                                                                                         item)

            start_pos = item["start"] + span_index

            # If other token
            other_token = line[i: start_pos]
            if len(other_token) > 0:
                tokens.append(other_token)
                tokens_labels.append(self.label_mapper.other_label)

            # Start entity
            end_pos = start_pos + length
            entity_token = line[start_pos: end_pos]
            tokens.append(entity_token)
            tokens_labels.append(self.label_mapper.standardise_label_name(anno_type))

            i = end_pos

        other_token = line[i:]
        if len(other_token) > 0:
            tokens.append(other_token)
            tokens_labels.append(self.label_mapper.other_label)

        return tokens, tokens_labels
