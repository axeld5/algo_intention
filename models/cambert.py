import evaluate
import os 
import torch
import warnings
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline 
from typing import List, Dict 

from .utils import ToTorchDataset, encode_labels
from .metrics import pure_accuracy, penalize_luggage_lost_errors, penalize_out_scope_errors

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

class BertModel:
    
    def __init__(self, num_labels:int=9, num_train_epochs:int=10) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels)
        self.training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=num_train_epochs)
        self.inv_label_dict = None 

    def fit(self, train_texts:List[str], train_labels:List[int], eval_texts:List[str], eval_labels:List[int]) -> None:
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.encode_data_(train_texts, train_labels),
            eval_dataset=self.encode_data_(eval_texts, eval_labels),
            compute_metrics=self.compute_metrics_,
        )
        trainer.train()

    def predict(self, texts:List[str]) -> List[int]:
        classifier = pipeline(task="text-classification", model=self.model.to("cpu"), tokenizer=self.tokenizer)
        pred_labels = [0]*len(texts) 
        for i in range(len(texts)):
            prediction = classifier(texts[i])
            label = int(prediction[0]['label'][-1])
            pred_labels[i] = label
        return pred_labels
    
    def load_model(self, path):
        self.model = self.model.from_pretrained(path)
        return self

    def evaluate_metrics(self, texts:List[str], vect_labels:List[int], label_dict:Dict[str, int]) -> Dict[str, float]:
        metric_dict = {}
        pred_labels = self.predict(texts)
        metric_dict["accuracy"] = pure_accuracy(vect_labels, pred_labels)
        metric_dict["lost_lug_pen"] = penalize_luggage_lost_errors(vect_labels, pred_labels, label_dict)
        metric_dict["out_scope_err"] = penalize_out_scope_errors(vect_labels, pred_labels, label_dict)
        return metric_dict
    
    def encode_data_(self, texts:List[str], labels:List[int]) -> torch.utils.data.Dataset:
        tokens = self.tokenizer(texts, padding=True)
        dataset = ToTorchDataset(tokens, labels)
        return dataset

    def compute_metrics_(self, eval_pred:List[int]) -> dict:
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
