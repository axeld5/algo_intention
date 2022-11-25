import argparse 
import json
import pandas as pd 

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from typing import List, Dict

from utils import vectorize_data, invert_label_dictionary
from vis import visualize
from models.ml_model import MLModel
from models.cambert import BertModel

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

def train_and_save(data:pd.DataFrame, bert_model:BertModel, ml_model_dict:List[MLModel], random_state:int=42) -> None:
    texts = data['text'].tolist()
    vect_texts, vect_labels, label_dict = vectorize_data(args.filename)    
    train_bert_texts, eval_texts, train_bert_labels, eval_labels = train_test_split(texts, vect_labels, test_size=0.2, random_state=random_state)
    model_perf = {}
    bert_model.fit(train_bert_texts, train_bert_labels, eval_texts, eval_labels)
    model_perf["bert"] = bert_model.evaluate_metrics(texts, vect_labels, label_dict)
    for model_name, model in ml_model_dict.items():
        model.fit(vect_texts, vect_labels)
    bert_model.model.save_pretrained("./saved_models/bert_model")
    save_model(ml_model_dict)
    out_file = open('./saved_models/inv_label_dict.json','w+')
    json.dump(invert_label_dictionary(label_dict),out_file)


def save_model(ml_model_dict:List[MLModel]) -> None:
    for model_name, model in ml_model_dict.items():
        dump(model, "./saved_models/"+model_name+".joblib")
    
if __name__ == "__main__":
    data = pd.read_csv(args.filename)
    train_and_save(data, BertModel(num_train_epochs=10),
        {"random_forest" : MLModel(RandomForestClassifier()), 
        "log_regression": MLModel(LogisticRegressionCV())})