import argparse 
import json 
import pandas as pd 

from joblib import load

from models.ml_model import MLModel
from models.cambert import BertModel

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

if __name__ == "__main__":
    data = pd.read_csv(args.filename)
    texts = data['text'].tolist()
    bert_model = BertModel().load_model("./saved_models/bert_model")
    predicted_labels = bert_model.predict(texts)
    f = open("./saved_models/inv_label_dict.json",)
    inv_label_dict = json.load(f)
    for i in range(len(predicted_labels)):
        predicted_labels[i] = inv_label_dict[str(predicted_labels[i])][0]
    print(predicted_labels) 