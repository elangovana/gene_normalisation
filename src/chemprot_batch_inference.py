import argparse
import csv
import json
import logging
import os.path
import sys
import tarfile
from io import StringIO

from bert_ner_position_converter_include_space import BertNerPositionIncludeSpaceConverter
from datasets.chemprot_ner_label_mapper import ChemprotNerLabelMapper
from serve import model_fn, predict_fn


class ChemprotBatchInference:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def process_file(self, inputfile, model, outputfile_prefix, batch_size=8):
        self._logger.info("Processing file {}".format(inputfile))

        with open(inputfile, "r") as f:
            pubmed_docs = json.load(f)
            result = []
            for doc_batch in self._chunk(pubmed_docs, size=batch_size):
                result.extend(self._process_batch(doc_batch, model))
        self._logger.info("Completed inference file {}".format(inputfile))

        self._write_entities(result, f"{outputfile_prefix}.anon.txt")
        self._write_abstract(result, f"{outputfile_prefix}.abstract.tsv")

        self._logger.info("Completed processing file {}".format(inputfile))

    def _process_batch(self, doc_batch, model):
        texts = [d["article_abstract"] for d in doc_batch]
        batch_result = predict_fn(texts, model)
        result = []
        for br, d in zip(batch_result, doc_batch):
            result.append({"docid": d["pubmed_id"],
                           "text": d["article_abstract"],
                           "entities_detected": br})
        return result

    def process_dir(self, inputdir, modeltar, outputdir, batch_size=8):
        outputmodeldir = os.path.join(os.path.dirname(modeltar), "artifacts")
        self._extract_tar(modeltar, outputmodeldir)
        model = model_fn(outputmodeldir)
        for f in os.listdir(inputdir):
            self.process_file(os.path.join(inputdir, f), model, os.path.join(outputdir, f),
                              batch_size=batch_size)

    def _write_entities(self, json_result, outputfile):
        label_mapper = ChemprotNerLabelMapper()
        c = BertNerPositionIncludeSpaceConverter()
        c.process_file(StringIO(json.dumps(json_result)), outputfile, label_mapper.other_label,
                       label_mapper.entity_labels,
                       label_mapper.continuation_symbol)

    def _write_abstract(self, json_result, outputfile):
        with open(outputfile, "w") as f:
            csv_writer = csv.writer(f, delimiter='\t', quotechar=None)
            for d in json_result:
                csv_writer.writerow([d["docid"], d["text"].replace("\t", " ").replace("\n", " ").replace("\r", " ")])

    def _chunk(self, l, size=5):
        for i in range(0, len(l), size):
            yield l[i:i + size]

    def _extract_tar(self, modeltargz, outputmodeldir):
        with tarfile.open(modeltargz) as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, outputmodeldir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdatadir", required=True,
                        help="The input data  dir")

    parser.add_argument("--modeltar",
                        help="The model tar file", required=True)

    parser.add_argument("--outputdatadir",
                        help="The model tar file", required=True)

    parser.add_argument("--batchsize",
                        help="The batch size", required=False, type=int, default=8)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)
    ChemprotBatchInference().process_dir(args.inputdatadir, args.modeltar, args.outputdatadir, args.batchsize)


if "__main__" == __name__:
    main()
