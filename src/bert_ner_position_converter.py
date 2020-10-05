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
        for si, s in enumerate(entities_detected):

            if s["entity"] == other_label:
                if s["raw_token"] not in ('[CLS]'): offset += len(s["raw_token"].lstrip("#"))
                continue
            elif s["entity"] not in entity_labels:
                continue

            entity = s["raw_token"]
            for j in range(si + 1, len(entities_detected)):
                if entities_detected[j]["entity"] != continuation_symbol_dict[s["entity"]]: break
                raw_token = entities_detected[j]["raw_token"]
                sep = " "
                if raw_token.startswith("#") or raw_token == ".":
                    sep = ""

                entity = entity + sep + entities_detected[j]["raw_token"].lstrip("#")

            entity_npsp = entity.replace(" ", "")
            assert entity_npsp in original_text_npsp, "`{}` not in {}".format(entity, original_text_npsp)
            margin = 2
            approx_offset = offset - margin if offset > margin else 0
            approx_end = approx_offset + len(entity) + margin * 2
            count_occ = original_text_npsp.count(entity_npsp, approx_offset, approx_end)
            assert count_occ == 1, "{} - found `{}` n times {} in {} after offset {} {}".format(doc_id, entity_npsp,
                                                                                                count_occ,
                                                                                                original_text_npsp,
                                                                                                approx_offset,
                                                                                                approx_end)

            start_pos = original_text_npsp.find(entity_npsp, approx_offset, approx_end)
            end_pos = start_pos + len(entity_npsp) - 1
            result.append((doc_id, start_pos, end_pos, entity))
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
