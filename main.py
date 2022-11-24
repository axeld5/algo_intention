import argparse 
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from typing import List, Dict

from utils import vectorize_data
from vis import visualize
from models.ml_model import MLModel
from models.cambert import BertModel

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

def model_perf_eval(data:pd.DataFrame, bert_model:BertModel, ml_model_dict:List[MLModel]) -> Dict[str, Dict[str, int]]:
    texts = data['text'].tolist()
    vect_texts, vect_labels, label_dict = vectorize_data(args.filename)    
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, vect_labels, test_size=0.33, random_state=42)
    train_vect_texts, test_vect_texts, train_labels, test_labels = train_test_split(vect_texts, vect_labels, test_size=0.33, random_state=42)
    model_perf = {}
    bert_model.fit(train_texts, train_labels, test_texts, test_labels)
    model_perf["bert"] = bert_model.evaluate_metrics(test_texts, test_labels, label_dict)
    for model_name, model in ml_model_dict.items():
        model.fit(train_vect_texts, train_labels)
        model_perf[model_name] = model.evaluate_metrics(test_vect_texts, test_labels, label_dict)
    return model_perf 

if __name__ == "__main__":
    data = pd.read_csv(args.filename)    
    model_perf = model_perf_eval(data, BertModel(num_train_epochs=10), 
            {"random_forest" : MLModel(RandomForestClassifier()), 
            "log_regression": MLModel(LogisticRegressionCV())})
    visualize(model_perf)