import torch

from typing import List, Dict

def binarize_labels(labels:List[str], label:str) -> List[int]:
    encoded_labels = [0]*len(labels)
    for i in range(len(labels)):
        if labels[i] == label:
            encoded_labels[i] = 1
    return encoded_labels 

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