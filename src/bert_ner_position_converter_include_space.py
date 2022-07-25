import argparse
import json
import logging
import sys

from datasets.chemprot_ner_label_mapper import ChemprotNerLabelMapper


class BertNerPositionIncludeSpaceConverter:
    """

    """

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def locate_position(self, original_text, entities_detected, entity_labels, continuation_symbol_dict,
                        doc_id=""):
        token_offset = 0
        result = []
        prev_entity_type = None
        potential_entity_text = None
        entity_offset = None
        for entity_metadata in entities_detected:
            raw_token = entity_metadata["raw_token"]
            entity_type = entity_metadata["entity"]
            clean_token = raw_token
            if raw_token in ['[CLS]']: continue

            # White space...
            if token_offset > 0:
                if raw_token.startswith("#"):
                    # clean the token continuation
                    clean_token = clean_token[2:]
                else:
                    # Space
                    token_offset += 1

            # If continuation
            if raw_token.startswith("#"):
                if prev_entity_type and continuation_symbol_dict[prev_entity_type] == entity_type:
                    potential_entity_text = potential_entity_text + clean_token
                elif prev_entity_type and continuation_symbol_dict[prev_entity_type] != entity_type:
                    # Handle, case continuation symbol but the entity continuation doesnt match..
                    prev_entity_type = None
            elif prev_entity_type and continuation_symbol_dict[prev_entity_type] == entity_type:
                potential_entity_text = potential_entity_text + " " + clean_token
            elif entity_type in entity_labels:
                potential_entity_text = clean_token
                entity_offset = token_offset
                prev_entity_type = entity_type

            else:
                if prev_entity_type is not None:
                    end_pos = entity_offset + len(potential_entity_text)

                    if original_text[entity_offset: end_pos] != potential_entity_text:
                        self._logger.warning(
                            "Something went wrong..position at original text  `{}` doesnt line up with entity text `{}`. Skipping the rest of items in this record.. ".format(
                                original_text[entity_offset: end_pos], potential_entity_text))
                        return result

                    result.append((doc_id, prev_entity_type[2:], entity_offset, end_pos, potential_entity_text))
                prev_entity_type = None

            token_offset += len(clean_token)

        return result

    def process_file(self, input_file_or_handle, output_file_or_handler, other_label, entity_labels,
                     continuation_symbol_dict):
        data = self._load_json(input_file_or_handle)
        result = []
        for item in data:
            docid = item["docid"]
            raw_text = item["text"]
            entities_detected = item["entities_detected"]

            pos_result = self.locate_position(raw_text, entities_detected, entity_labels,
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
        if len(items) == 0:
            self._logger.info("No entities detected..Hence not writing output file")
            return

        s = ""
        for i_i, i in enumerate(items):
            s += "{}\tT{}\t{}\t{}\t{}\t{}\n".format(i[0], i_i, i[1], i[2], i[3], i[4])

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
    label_mapper = ChemprotNerLabelMapper()
    c = BertNerPositionIncludeSpaceConverter()
    c.process_file(args.inputfile, args.outputfile, label_mapper.other_label, label_mapper.entity_labels,
                   label_mapper.continuation_symbol)
