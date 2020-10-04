import glob
import json
import os
import pickle

import torch

from model.bert_model import BertModel

"""
This is the sagemaker inference entry script
"""
CSV_CONTENT_TYPE = 'text/csv'
JSON_CONTENT_TYPE = 'text/json'


def model_fn(model_dir):
    # Load label mapper
    label_mapper_pickle_file = os.path.join(model_dir, "label_mapper.pkl")
    with open(label_mapper_pickle_file, "rb") as f:
        label_mapper = pickle.load(f)

    # Load model
    device = get_device()
    model = BertModel(model_dir, pretrained_num_classes=label_mapper.num_classes)
    model.to(device=device)

    # Load preprocessor
    preprocessor_pickle_file = os.path.join(model_dir, "preprocessor.pkl")
    with open(preprocessor_pickle_file, "rb") as f:
        preprocessor_mapper = pickle.load(f)

    return preprocessor_mapper, model, label_mapper


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def input_fn(input, content_type):
    if content_type == CSV_CONTENT_TYPE:
        records = input.split("\n")
        return records
    else:
        raise ValueError(
            "Content type {} not supported. The supported type is {}".format(content_type, CSV_CONTENT_TYPE))


def preprocess(input_batch, preprocessor):
    result = [preprocessor([i])[0].unsqueeze(dim=0) for i in input_batch]
    result = torch.cat(result)
    return result


def convert_tokens_to_words(batch_raw_tokens, predictions):
    result = []
    for i, s_raw_tokens in enumerate(batch_raw_tokens):
        r = []
        raw_token_len = len(s_raw_tokens)
        pred_row = predictions[i]

        pred_len = len(pred_row)
        assert raw_token_len == pred_len, "The length of the tokens {} and the results {} do not match".format(
            raw_token_len, pred_len)

        for t in range(raw_token_len):
            entity_begin = pred_row[t]["entity"]

            probability = pred_row[t]["probablity"]

            token = s_raw_tokens[t]
            r.append({"raw_token": token, "probability": probability, "entity": entity_begin})

        result.append(r)
    return result


def predict_fn(input, model_artifacts):
    preprocessor, model, label_mapper = model_artifacts

    # Pre-process
    input_tensor = preprocess(input, preprocessor)

    # tokens
    batch_raw_tokens = [preprocessor.pad(preprocessor.tokenise([i])[0])[0] for i in input]

    # Copy input to gpu if available
    device = get_device()
    input_tensor = input_tensor.to(device=device)

    # Invoke
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
        # Convert to probabilities
        softmax = torch.nn.Softmax(dim=2)
        output_tensor = softmax(output_tensor)

    # Return the class with the highest prob and the corresponding prob
    prob, class_indices = torch.max(output_tensor, dim=2)
    result = []
    for c_seq, p_seq in zip(class_indices, prob):
        r = []
        for ci, pi in zip(c_seq, p_seq):
            r.append({"entity": label_mapper.index_to_label(ci.item()), "probablity": pi.item()})
        result.append(r)

    result = convert_tokens_to_words(batch_raw_tokens, result)

    return result


def output_fn(output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        prediction = json.dumps(output)
        return prediction, accept
    else:
        raise ValueError(
            "Content type {} not supported. The only types supported are {}".format(accept, JSON_CONTENT_TYPE))

#
# if __name__ == '__main__':
#
#     model = model_fn("../tmp")
#     from datasets.biocreative_dataset import BiocreativeDataset
#
#     d = BiocreativeDataset("../tmp/train.in", None)
#     for i in range( len(d)):
#
#         input = input_fn("".join(d[i][0]), "text/csv")
#         r = predict_fn(input, model)
#         for i in r[0]:
#             if i["entity"] != "O":
#                 print(i["entity"], i["raw_token"])
#
#         break
