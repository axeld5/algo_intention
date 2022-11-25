import torch

from typing import List, Dict

def binarize_labels(labels:List[int], label:int) -> List[int]:
    encoded_labels = [0]*len(labels)
    for i in range(len(labels)):
        if labels[i] == label:
            encoded_labels[i] = 1
    return encoded_labels 

def invert_label_dictionary(label_dict):
    inverted_label_dict = {} 
    for k, v in label_dict.items():
        inverted_label_dict[v] = k
    return inverted_label_dict

class ToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings:Dict[str, int], labels:List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx:int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)