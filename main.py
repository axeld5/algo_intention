import argparse 
import json
import pandas as pd 

from copy import copy 
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

def model_perf_eval(data:pd.DataFrame, bert_model:BertModel, ml_model_dict:List[MLModel], random_state:int=42) -> Dict[str, Dict[str, int]]:
    texts = data['text'].tolist()
    vect_texts, vect_labels, label_dict = vectorize_data(args.filename)    
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, vect_labels, test_size=0.33, random_state=random_state)
    train_bert_texts, eval_texts, train_bert_labels, eval_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=random_state)
    train_vect_texts, test_vect_texts, train_labels, test_labels = train_test_split(vect_texts, vect_labels, test_size=0.33, random_state=random_state)
    model_perf = {}
    bert_model.fit(train_bert_texts, train_bert_labels, eval_texts, eval_labels)
    model_perf["bert"] = bert_model.evaluate_metrics(test_texts, test_labels, label_dict)
    for model_name, model in ml_model_dict.items():
        model.fit(train_vect_texts, train_labels)
        model_perf[model_name] = model.evaluate_metrics(test_vect_texts, test_labels, label_dict)
    bert_model.model.save_pretrained("./saved_models/")
    save_model(ml_model_dict)
    out_file = open('./saved_models/inv_label_dict.json','w+')
    json.dump(invert_label_dictionary(label_dict),out_file)
    return model_perf

def average_perf_eval(data:pd.DataFrame, bert_model:BertModel, ml_model_dict:List[MLModel], rstate_list:List[int]) -> Dict[str, Dict[str, int]]:
    n_iter = len(rstate_list)
    model_perf_list = []
    for i in range(n_iter):
        bert_model_copy = copy(bert_model)
        ml_model_dict_copy = copy(ml_model_dict)
        model_perf_list.append(model_perf_eval(data, bert_model_copy, ml_model_dict_copy, rstate_list[i]))
    final_perf = model_perf_list[0]
    for i in range(1, n_iter):
        for model_name, performances in model_perf_list[i].items():
            for metric, score in performances.items():
                final_perf[model_name][metric] += score 
    for model_name, performances in final_perf.items():
        for metric, score in performances.items():
            final_perf[model_name][metric] = final_perf[model_name][metric]/n_iter
    return final_perf

def save_model(ml_model_dict:List[MLModel]) -> None:
    for model_name, model in ml_model_dict.items():
        dump(model, "./saved_models/bert_model"+model_name+".joblib")


if __name__ == "__main__":
    task = "save"
    if task == "save":
        data = pd.read_csv(args.filename)
        model_perf = model_perf_eval(data, BertModel(num_train_epochs=1),
                {"random_forest" : MLModel(RandomForestClassifier()), 
                "log_regression": MLModel(LogisticRegressionCV())})
        visualize(model_perf)
    if task == "compare":
        data = pd.read_csv(args.filename)    
        rstate_list = [i for i in range(10)]
        n_iter = len(rstate_list)
        model_perf_list = []
        for i in range(n_iter):
            model_perf_list.append(model_perf_eval(data, BertModel(num_train_epochs=10), 
                {"random_forest" : MLModel(RandomForestClassifier()), 
                "log_regression": MLModel(LogisticRegressionCV())}, rstate_list[i]))
        final_perf = model_perf_list[0]
        for i in range(1, n_iter):
            for model_name, performances in model_perf_list[i].items():
                for metric, score in performances.items():
                    final_perf[model_name][metric] += score 
        for model_name, performances in final_perf.items():
            for metric, score in performances.items():
                final_perf[model_name][metric] = final_perf[model_name][metric]/n_iter
        visualize(final_perf)