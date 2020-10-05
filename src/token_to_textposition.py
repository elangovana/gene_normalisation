import json

from datasets.biocreative_ner_label_mapper import BiocreativeNerLabelMapper


class TokenToTextPosition:

    def locate_position(self, original_text, entities_detected, other_label, entity_labels, continuation_symbol_dict,
                        doc_id=""):
        offset = 0
        result = []
        for si, s in enumerate(entities_detected):

            if s["entity"] == other_label:
                if s["raw_token"] not in ('[CLS]'):
                    space = 1
                    if s["raw_token"].startswith("#") or s["raw_token"] == ".":
                        space = 0
                    offset = offset + space + len(s["raw_token"].lstrip("#"))
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

            entity = entity.replace(" - ", "-")

            assert entity in original_text, "`{}` not in {}".format(entity, original_text)
            margin = 6
            approx_offset = offset - margin if offset > margin else 0
            approx_end = approx_offset + len(entity) + margin * 2
            count_occ = original_text.count(entity, approx_offset, approx_end)
            assert count_occ == 1, "{} - found `{}` n times {} in {} after offset {} {}".format(doc_id, entity,
                                                                                                count_occ,
                                                                                                original_text,
                                                                                                approx_offset,
                                                                                                approx_end)

            start_pos = original_text.find(entity, approx_offset, approx_end)
            end_pos = start_pos + len(entity)
            result.append((doc_id, start_pos, end_pos, entity))
            offset = start_pos + len(entity) + 1
        return result


if __name__ == '__main__':
    results_file = "../result_sample_train.json"
    with open(results_file) as f:
        data = json.load(f)
    label_mapper = BiocreativeNerLabelMapper()
    for item in data:
        docid = item["docid"]
        raw_text = item["text"]
        entities_detected = item["entities_detected"]

        result = TokenToTextPosition().locate_position(raw_text, entities_detected, label_mapper.other_label,
                                                       label_mapper.entity_labels, label_mapper.continuation_symbol,
                                                       docid)
        if len(result) > 0:
            print(result)
