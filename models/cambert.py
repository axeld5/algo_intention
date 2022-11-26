import evaluate
import os 
import torch
import warnings
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline 
from typing import List, Dict, Tuple

from .utils import ToTorchDataset
from .metrics import pure_accuracy, penalize_luggage_lost_errors, penalize_out_scope_errors

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

class BertModel:
    
    def __init__(self, num_labels:int=9, num_train_epochs:int=10, random_state:int=42) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels)
        self.training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=num_train_epochs)
        self.encoder = LabelEncoder()
        self.random_state = random_state

    def fit(self, texts:List[str], labels:List[str]) -> None:
        encoded_labels = self.encode_labels_(labels)
        train_dataset, eval_dataset = self.encode_data_(texts, encoded_labels)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics_,
        )
        trainer.train()

    def predict(self, texts:List[str]) -> List[str]:
        classifier = pipeline(task="text-classification", model=self.model.to("cpu"), tokenizer=self.tokenizer)
        pred_labels = [0]*len(texts) 
        for i in range(len(texts)):
            prediction = classifier(texts[i])
            label = int(prediction[0]['label'][-1])
            pred_labels[i] = label
        pred_labels = self.encoder.inverse_transform(pred_labels).tolist()
        return pred_labels
    
    def load_model(self, path):
        self.model = self.model.from_pretrained(path)
        return self

    def evaluate_metrics(self, texts:List[str], labels:List[str]) -> Dict[str, float]:
        metric_dict = {}
        pred_labels = self.predict(texts)
        metric_dict["accuracy"] = pure_accuracy(labels, pred_labels)
        metric_dict["lost_lug_pen"] = penalize_luggage_lost_errors(labels, pred_labels)
        metric_dict["out_scope_err"] = penalize_out_scope_errors(labels, pred_labels)
        return metric_dict
    
    def encode_data_(self, texts:List[str], labels:List[int]) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2, random_state=self.random_state)
        train_tokens = self.tokenizer(train_texts, padding=True)
        eval_tokens = self.tokenizer(eval_texts, padding=True)
        train_dataset = ToTorchDataset(train_tokens, train_labels)
        eval_dataset = ToTorchDataset(eval_tokens, eval_labels)
        return train_dataset, eval_dataset
    
    def encode_labels_(self, labels:List[str]) -> np.ndarray:
        vect_labels = self.encoder.fit_transform(labels)
        return vect_labels

    def compute_metrics_(self, eval_pred:List[int]) -> dict:
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
