import json
from io import StringIO
from unittest import TestCase

from bert_ner_position_converter import BertNerPositionConverter


class TestBertNerPositionConverter(TestCase):

    def test_locate_position(self):
        # Arrange
        sut = BertNerPositionConverter()
        input_raw = "Comparison with alkaline phosphatases and 5-nucleotidase"
        docid = 1

        expected = [
            (1, 14, 33, 'alkaline phosphatases')
            , (1, 37, 50, '5 - nucleotidase')
        ]
        entities_detected = [{"raw_token": "[CLS]", "probability": 0.999981164932251, "entity": "O"},
                             {"raw_token": "Co", "probability": 0.9999797344207764, "entity": "O"},
                             {"raw_token": "##mp", "probability": 0.9999808073043823, "entity": "O"},
                             {"raw_token": "##aris", "probability": 0.9999829530715942, "entity": "O"},
                             {"raw_token": "##on", "probability": 0.9999872446060181, "entity": "O"},
                             {"raw_token": "with", "probability": 0.9999898672103882, "entity": "O"},
                             {"raw_token": "al", "probability": 0.9995115995407104, "entity": "B-GENE"},
                             {"raw_token": "##kal", "probability": 0.999380350112915, "entity": "I-GENE"},
                             {"raw_token": "##ine", "probability": 0.9998010993003845, "entity": "I-GENE"},
                             {"raw_token": "p", "probability": 0.999666690826416, "entity": "I-GENE"},
                             {"raw_token": "##hos", "probability": 0.9998422861099243, "entity": "I-GENE"},
                             {"raw_token": "##pha", "probability": 0.9998385906219482, "entity": "I-GENE"},
                             {"raw_token": "##tase", "probability": 0.9996901750564575, "entity": "I-GENE"},
                             {"raw_token": "##s", "probability": 0.9996976852416992, "entity": "I-GENE"},
                             {"raw_token": "and", "probability": 0.9999517202377319, "entity": "O"},
                             {"raw_token": "5", "probability": 0.9963992834091187, "entity": "B-GENE"},
                             {"raw_token": "-", "probability": 0.9996247291564941, "entity": "I-GENE"},
                             {"raw_token": "n", "probability": 0.9990386962890625, "entity": "I-GENE"},
                             {"raw_token": "##uc", "probability": 0.9998058676719666, "entity": "I-GENE"},
                             {"raw_token": "##leo", "probability": 0.999792754650116, "entity": "I-GENE"},
                             {"raw_token": "##ti", "probability": 0.9998144507408142, "entity": "I-GENE"},
                             {"raw_token": "##das", "probability": 0.9994984865188599, "entity": "I-GENE"},
                             {"raw_token": "##e", "probability": 0.9997034668922424, "entity": "I-GENE"},
                             {"raw_token": "[PAD]", "probability": 0.9996922016143799, "entity": "O"},
                             {"raw_token": "[PAD]", "probability": 0.9999816417694092, "entity": "O"}]

        # Act
        actual = sut.locate_position(input_raw, entities_detected, "O", "B-GENE", {"B-GENE": "I-GENE"}, docid)

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test_locate_position_repeated_entity(self):
        # Arrange
        sut = BertNerPositionConverter()
        input_raw = "growth arrest in G(1), no growth defect was observed in a double cwh43 cln3 mutants."
        docid = 1

        expected = [
            (1, 53, 57, 'cwh43'),
            (1, 58, 68, 'cln3 mutants')
        ]
        entities_detected = [{'raw_token': '[CLS]', 'probability': 0.9999691247940063, 'entity': 'O'},
                             {'raw_token': 'growth', 'probability': 0.9998800754547119, 'entity': 'O'},
                             {'raw_token': 'arrest', 'probability': 0.9998175501823425, 'entity': 'O'},
                             {'raw_token': 'in', 'probability': 0.9993170499801636, 'entity': 'O'},
                             {'raw_token': 'G', 'probability': 0.8757198452949524, 'entity': 'O'},
                             {'raw_token': '(', 'probability': 0.9296509623527527, 'entity': 'I-GENE'},
                             {'raw_token': '1', 'probability': 0.9153020977973938, 'entity': 'I-GENE'},
                             {'raw_token': ')', 'probability': 0.8155266046524048, 'entity': 'I-GENE'},
                             {'raw_token': ',', 'probability': 0.9999626874923706, 'entity': 'O'},
                             {'raw_token': 'no', 'probability': 0.9999707937240601, 'entity': 'O'},
                             {'raw_token': 'growth', 'probability': 0.9999169111251831, 'entity': 'O'},
                             {'raw_token': 'defect', 'probability': 0.999941349029541, 'entity': 'O'},
                             {'raw_token': 'was', 'probability': 0.9999703168869019, 'entity': 'O'},
                             {'raw_token': 'observed', 'probability': 0.9999804496765137, 'entity': 'O'},
                             {'raw_token': 'in', 'probability': 0.9999634027481079, 'entity': 'O'},
                             {'raw_token': 'a', 'probability': 0.9999642372131348, 'entity': 'O'},
                             {'raw_token': 'double', 'probability': 0.961931049823761, 'entity': 'O'},
                             {'raw_token': 'c', 'probability': 0.9993970394134521, 'entity': 'B-GENE'},
                             {'raw_token': '##w', 'probability': 0.9997829794883728, 'entity': 'I-GENE'},
                             {'raw_token': '##h', 'probability': 0.9997020363807678, 'entity': 'I-GENE'},
                             {'raw_token': '##43', 'probability': 0.9995094537734985, 'entity': 'I-GENE'},
                             {'raw_token': 'c', 'probability': 0.9588805437088013, 'entity': 'B-GENE'},
                             {'raw_token': '##ln', 'probability': 0.9997311234474182, 'entity': 'I-GENE'},
                             {'raw_token': '##3', 'probability': 0.9994176626205444, 'entity': 'I-GENE'},
                             {'raw_token': 'mutant', 'probability': 0.8578420877456665, 'entity': 'I-GENE'},
                             {'raw_token': '##s', 'probability': 0.594409704208374, 'entity': 'I-GENE'},
                             {'raw_token': '.', 'probability': 0.999903678894043, 'entity': 'O'}
                             ]

        # Act
        actual = sut.locate_position(input_raw, entities_detected, "O", "B-GENE", {"B-GENE": "I-GENE"}, docid)

        # Assert

        self.assertSequenceEqual(expected, actual)

    def test_process_file(self):
        # Arrange
        input_raw = "Comparison with alkaline phosphatases and 5-nucleotidase"
        docid = "DOC1"
        expected = """DOC1|14 33|alkaline phosphatases
DOC1|37 50|5 - nucleotidase
"""
        entities_detected = [{"raw_token": "[CLS]", "probability": 0.999981164932251, "entity": "O"},
                             {"raw_token": "Co", "probability": 0.9999797344207764, "entity": "O"},
                             {"raw_token": "##mp", "probability": 0.9999808073043823, "entity": "O"},
                             {"raw_token": "##aris", "probability": 0.9999829530715942, "entity": "O"},
                             {"raw_token": "##on", "probability": 0.9999872446060181, "entity": "O"},
                             {"raw_token": "with", "probability": 0.9999898672103882, "entity": "O"},
                             {"raw_token": "al", "probability": 0.9995115995407104, "entity": "B-GENE"},
                             {"raw_token": "##kal", "probability": 0.999380350112915, "entity": "I-GENE"},
                             {"raw_token": "##ine", "probability": 0.9998010993003845, "entity": "I-GENE"},
                             {"raw_token": "p", "probability": 0.999666690826416, "entity": "I-GENE"},
                             {"raw_token": "##hos", "probability": 0.9998422861099243, "entity": "I-GENE"},
                             {"raw_token": "##pha", "probability": 0.9998385906219482, "entity": "I-GENE"},
                             {"raw_token": "##tase", "probability": 0.9996901750564575, "entity": "I-GENE"},
                             {"raw_token": "##s", "probability": 0.9996976852416992, "entity": "I-GENE"},
                             {"raw_token": "and", "probability": 0.9999517202377319, "entity": "O"},
                             {"raw_token": "5", "probability": 0.9963992834091187, "entity": "B-GENE"},
                             {"raw_token": "-", "probability": 0.9996247291564941, "entity": "I-GENE"},
                             {"raw_token": "n", "probability": 0.9990386962890625, "entity": "I-GENE"},
                             {"raw_token": "##uc", "probability": 0.9998058676719666, "entity": "I-GENE"},
                             {"raw_token": "##leo", "probability": 0.999792754650116, "entity": "I-GENE"},
                             {"raw_token": "##ti", "probability": 0.9998144507408142, "entity": "I-GENE"},
                             {"raw_token": "##das", "probability": 0.9994984865188599, "entity": "I-GENE"},
                             {"raw_token": "##e", "probability": 0.9997034668922424, "entity": "I-GENE"},
                             {"raw_token": "[PAD]", "probability": 0.9996922016143799, "entity": "O"},
                             {"raw_token": "[PAD]", "probability": 0.9999816417694092, "entity": "O"}]

        input_json = [{"docid": docid, "text": input_raw, "entities_detected": entities_detected}]

        input_handle = StringIO(json.dumps(input_json))
        output_handle_actual = StringIO()

        sut = BertNerPositionConverter()

        # Act
        sut.process_file(input_handle, output_handle_actual, "O", "B-GENE", {"B-GENE": "I-GENE"})

        # Assert
        self.assertEqual(expected, output_handle_actual.getvalue())
