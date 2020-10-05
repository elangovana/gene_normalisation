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

    def test_locate_position_entity_last_token(self):
        # Arrange
        sut = BertNerPositionConverter()
        input_raw = "growth arrest inG"
        docid = 1

        expected = [
            (1, 14, 14, 'G')

        ]

        entities_detected = [{'raw_token': '[CLS]', 'probability': 0.9999691247940063, 'entity': 'O'},
                             {'raw_token': 'growth', 'probability': 0.9998800754547119, 'entity': 'O'},
                             {'raw_token': 'arrest', 'probability': 0.9998175501823425, 'entity': 'O'},
                             {'raw_token': 'in', 'probability': 0.9993170499801636, 'entity': 'O'},
                             {'raw_token': '##G', 'probability': 0.8757198452949524, 'entity': 'B-GENE'}]
        # Act
        actual = sut.locate_position(input_raw, entities_detected, "O", "B-GENE", {"B-GENE": "I-GENE"}, docid)

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test_locate_position_case2(self):
        sut = BertNerPositionConverter()
        docid = "DOC1"
        expected = [
            ('DOC1', 37, 41, 'Smad3')
            , ('DOC1', 43, 47, 'Smad4')
            , ('DOC1', 171, 173, 'AP1')
        ]

        input_raw = "Taken together, these data suggest that the Smad3/Smad4 complex has at least two separable nuclear functions: it forms a rapid, yet transient sequence-specific DNA binding complex, and it potentiates AP1-dependent transcriptional activation."
        entities_detected = [{'raw_token': '[CLS]', 'probability': 0.9999759197235107, 'entity': 'O'},
                             {'raw_token': 'Take', 'probability': 0.999964714050293, 'entity': 'O'},
                             {'raw_token': '##n', 'probability': 0.9999703168869019, 'entity': 'O'},
                             {'raw_token': 'together', 'probability': 0.9999674558639526, 'entity': 'O'},
                             {'raw_token': ',', 'probability': 0.9999845027923584, 'entity': 'O'},
                             {'raw_token': 'these', 'probability': 0.9999836683273315, 'entity': 'O'},
                             {'raw_token': 'data', 'probability': 0.9999768733978271, 'entity': 'O'},
                             {'raw_token': 'suggest', 'probability': 0.9999716281890869, 'entity': 'O'},
                             {'raw_token': 'that', 'probability': 0.9999637603759766, 'entity': 'O'},
                             {'raw_token': 'the', 'probability': 0.9999173879623413, 'entity': 'O'},
                             {'raw_token': 'S', 'probability': 0.9997923970222473, 'entity': 'B-GENE'},
                             {'raw_token': '##mad', 'probability': 0.9998654127120972, 'entity': 'I-GENE'},
                             {'raw_token': '##3', 'probability': 0.9998568296432495, 'entity': 'I-GENE'},
                             {'raw_token': '/', 'probability': 0.999716579914093, 'entity': 'O'},
                             {'raw_token': 'S', 'probability': 0.9997801184654236, 'entity': 'B-GENE'},
                             {'raw_token': '##mad', 'probability': 0.9997991919517517, 'entity': 'I-GENE'},
                             {'raw_token': '##4', 'probability': 0.9998315572738647, 'entity': 'I-GENE'},
                             {'raw_token': 'complex', 'probability': 0.999302864074707, 'entity': 'O'},
                             {'raw_token': 'has', 'probability': 0.9999722242355347, 'entity': 'O'},
                             {'raw_token': 'at', 'probability': 0.999977707862854, 'entity': 'O'},
                             {'raw_token': 'least', 'probability': 0.9999630451202393, 'entity': 'O'},
                             {'raw_token': 'two', 'probability': 0.9999375343322754, 'entity': 'O'},
                             {'raw_token': 'se', 'probability': 0.9999256134033203, 'entity': 'O'},
                             {'raw_token': '##par', 'probability': 0.999923586845398, 'entity': 'O'},
                             {'raw_token': '##able', 'probability': 0.9999620914459229, 'entity': 'O'},
                             {'raw_token': 'nuclear', 'probability': 0.9999343156814575, 'entity': 'O'},
                             {'raw_token': 'functions', 'probability': 0.9999608993530273, 'entity': 'O'},
                             {'raw_token': ':', 'probability': 0.9999692440032959, 'entity': 'O'},
                             {'raw_token': 'it', 'probability': 0.9999351501464844, 'entity': 'O'},
                             {'raw_token': 'forms', 'probability': 0.9999544620513916, 'entity': 'O'},
                             {'raw_token': 'a', 'probability': 0.9999556541442871, 'entity': 'O'},
                             {'raw_token': 'rapid', 'probability': 0.9998899698257446, 'entity': 'O'},
                             {'raw_token': ',', 'probability': 0.9999654293060303, 'entity': 'O'},
                             {'raw_token': 'yet', 'probability': 0.9999579191207886, 'entity': 'O'},
                             {'raw_token': 'trans', 'probability': 0.9999544620513916, 'entity': 'O'},
                             {'raw_token': '##ient', 'probability': 0.9999707937240601, 'entity': 'O'},
                             {'raw_token': 'sequence', 'probability': 0.9998983144760132, 'entity': 'O'},
                             {'raw_token': '-', 'probability': 0.999916672706604, 'entity': 'O'},
                             {'raw_token': 'specific', 'probability': 0.9999512434005737, 'entity': 'O'},
                             {'raw_token': 'DNA', 'probability': 0.9999529123306274, 'entity': 'O'},
                             {'raw_token': 'binding', 'probability': 0.9999488592147827, 'entity': 'O'},
                             {'raw_token': 'complex', 'probability': 0.999947190284729, 'entity': 'O'},
                             {'raw_token': ',', 'probability': 0.9999788999557495, 'entity': 'O'},
                             {'raw_token': 'and', 'probability': 0.9999576807022095, 'entity': 'O'},
                             {'raw_token': 'it', 'probability': 0.9999254941940308, 'entity': 'O'},
                             {'raw_token': 'potent', 'probability': 0.9999325275421143, 'entity': 'O'},
                             {'raw_token': '##iate', 'probability': 0.9999257326126099, 'entity': 'O'},
                             {'raw_token': '##s', 'probability': 0.9999498128890991, 'entity': 'O'},
                             {'raw_token': 'AP', 'probability': 0.9984390139579773, 'entity': 'B-GENE'},
                             {'raw_token': '##1', 'probability': 0.9993322491645813, 'entity': 'I-GENE'},
                             {'raw_token': '-', 'probability': 0.9995315074920654, 'entity': 'O'},
                             {'raw_token': 'dependent', 'probability': 0.9997039437294006, 'entity': 'O'},
                             {'raw_token': 'transcription', 'probability': 0.9997454285621643, 'entity': 'O'},
                             {'raw_token': '##al', 'probability': 0.9996950626373291, 'entity': 'O'},
                             {'raw_token': 'activation', 'probability': 0.9998255372047424, 'entity': 'O'},
                             {'raw_token': '.', 'probability': 0.9999220371246338, 'entity': 'O'}
                             ]

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
