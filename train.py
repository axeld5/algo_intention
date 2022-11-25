import argparse 
import json
import pandas as pd 

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from typing import List, Dict, Any

from models.ml_model import MLModel
from models.cambert import BertModel

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

def train_and_save(data:pd.DataFrame, ml_model_dict:Dict[str, Any]) -> None:
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    for model_name, model in ml_model_dict.items():
        model.fit(texts, labels)
    save_model(ml_model_dict)


def save_model(ml_model_dict:List[MLModel]) -> None:
    for model_name, model in ml_model_dict.items():
        dump(model, "./saved_models/"+model_name+".joblib")
    
if __name__ == "__main__":
    data = pd.read_csv(args.filename)
    train_and_save(data, 
        {"bert_model": BertModel(num_train_epochs=12),
        "random_forest" : MLModel(RandomForestClassifier()), 
        "log_regression": MLModel(LogisticRegressionCV())})