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
