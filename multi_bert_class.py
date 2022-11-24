import numpy as np
import pandas as pd 
import evaluate 
import torch

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline 

from vectorization import encode_labels
from metrics import pure_accuracy, penalize_luggage_lost_errors, penalize_out_scope_errors

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class ToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    data = pd.read_csv("data/intent-detection-train.csv")    
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    texts = data['text'].tolist()
    labels, label_dict = encode_labels(data['label'].tolist())
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.33)
    train_tokens = tokenizer(train_texts, padding=True)
    test_tokens = tokenizer(test_texts, padding=True)
    train_dataset = ToTorchDataset(train_tokens, train_labels)
    eval_dataset = ToTorchDataset(test_tokens, test_labels)
    model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=9)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=10)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
    pred_labels = [0]*len(test_labels) 
    for i in range(len(test_texts)):
        prediction = classifier(test_texts[i])
        label = int(prediction[0]['label'][-1])
        pred_labels[i] = label
    print(pure_accuracy(test_labels, pred_labels))
    print(penalize_out_scope_errors(test_labels, pred_labels, label_dict))
    print(penalize_luggage_lost_errors(test_labels, pred_labels, label_dict))
    