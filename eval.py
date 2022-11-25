import argparse 
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from typing import Dict, Any 

from vis import visualize
from models.ml_model import MLModel
from models.cambert import BertModel

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

def model_perf_eval(data:pd.DataFrame, ml_model_dict:Dict[str, Any], random_state:int=42) -> Dict[str, Dict[str, int]]:
    texts = data['text'].tolist()
    labels = data['label'].tolist()  
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.33, random_state=random_state)
    model_perf = {}
    for model_name, model in ml_model_dict.items():
        model.fit(train_texts, train_labels)
        model_perf[model_name] = model.evaluate_metrics(test_texts, test_labels)
    return model_perf

if __name__ == "__main__":
    data = pd.read_csv(args.filename)    
    rstate_list = [i for i in range(10)]
    n_iter = len(rstate_list)
    model_perf_list = []
    for i in range(n_iter):
        model_perf_list.append(model_perf_eval(data,  
            {"bert_model": BertModel(num_train_epochs=12, random_state=rstate_list[i]),
            "random_forest" : MLModel(RandomForestClassifier()), 
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