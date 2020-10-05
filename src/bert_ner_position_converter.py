import argparse
import json
import logging
import sys

from datasets.biocreative_ner_label_mapper import BiocreativeNerLabelMapper


class BertNerPositionConverter:
    """
    Converts the NER results from BioBERT to the Position in the original text defined as:

    The start-offset is the number of non-whitespace characters in the sentence preceding the first character of the mention,
    and the end-offset is the number of non-whitespace characters in the sentence preceding the last character of the mention.

    """

    def locate_position(self, original_text, entities_detected, other_label, entity_labels, continuation_symbol_dict,
                        doc_id=""):
        offset = 0
        result = []
        original_text_npsp = original_text.replace(" ", "")
        ner_i = 0
        while ner_i < len(entities_detected):
            ner_detection = entities_detected[ner_i]
            raw_token = ner_detection["raw_token"]
            entity_type = ner_detection["entity"]
            ner_i += 1

            if raw_token in ('[CLS]'): continue

            if entity_type not in entity_labels:
                offset = offset + len(raw_token.lstrip("#"))
                continue

            entity_text = raw_token
            # continue to search for the next parts of the entity
            while ner_i < len(entities_detected):
                continue_ner_detection = entities_detected[ner_i]
                ner_i += 1

                if continue_ner_detection["entity"] != continuation_symbol_dict[entity_type]: break
                raw_token = continue_ner_detection["raw_token"]
                sep = " "
                if raw_token.startswith("#") or raw_token == ".":
                    sep = ""

                entity_text = entity_text + sep + raw_token.lstrip("#")

            entity_npsp = entity_text.replace(" ", "")
            assert entity_npsp in original_text_npsp, "`{}` not in {}".format(entity_text, original_text_npsp)
            margin = 1
            approx_offset = offset - margin if offset > margin else 0
            approx_end = approx_offset + len(entity_npsp) + margin * 2
            count_occ = original_text_npsp.count(entity_npsp, approx_offset, approx_end)
            assert count_occ == 1, "{} - found `{}` n times {} in {} after offset {} {}".format(doc_id, entity_npsp,
                                                                                                count_occ,
                                                                                                original_text_npsp,
                                                                                                approx_offset,
                                                                                                approx_end)

            start_pos = original_text_npsp.find(entity_npsp, approx_offset, approx_end)
            end_pos = start_pos + len(entity_npsp) - 1
            result.append((doc_id, start_pos, end_pos, entity_text))

            ner_i = ner_i - 1
            offset = start_pos + len(entity_npsp)

        return result

    def process_file(self, input_file_or_handle, output_file_or_handler, other_label, entity_labels,
                     continuation_symbol_dict):
        data = self._load_json(input_file_or_handle)
        result = []
        for item in data:
            docid = item["docid"]
            raw_text = item["text"]
            entities_detected = item["entities_detected"]

            pos_result = self.locate_position(raw_text, entities_detected, other_label, entity_labels,
                                              continuation_symbol_dict,
                                              docid)
            result.extend(pos_result)

        self._write_file(result, output_file_or_handler)

    def _load_json(self, input_file_or_handle):
        if isinstance(input_file_or_handle, str):
            with open(input_file_or_handle, "r") as f:
                data = json.load(f)
        else:
            data = json.load(input_file_or_handle)
        return data

    def _write_file(self, items, input_file_or_handle):
        s = ""
        for i in items:
            s += "{}|{} {}|{}\n".format(i[0], i[1], i[2], i[3])

        if isinstance(input_file_or_handle, str):
            with open(input_file_or_handle, "w") as f:
                f.write(s)
        else:
            input_file_or_handle.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile",
                        help="The input file", required=True)
    parser.add_argument("--outputfile",
                        help="The outputfile", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)
    label_mapper = BiocreativeNerLabelMapper()
    c = BertNerPositionConverter()
    c.process_file(args.inputfile, args.outputfile, label_mapper.other_label, label_mapper.entity_labels,
                   label_mapper.continuation_symbol)
