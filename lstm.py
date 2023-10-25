import torch, torch.nn as nn
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

training_data = []
with open('data/train.jsonl', 'r') as f:
    datalist = list(f)
    for data in datalist:
        training_data.append(json.loads(data))

test_data = []
with open('data/test.jsonl', 'r') as f:
    datalist = list(f)
    for data in datalist:
        test_data.append(json.loads(data))

valid_data = []
with open('data/valid.jsonl', 'r') as f:
    datalist = list(f)
    for data in datalist:
        valid_data.append(json.loads(data))

class DNDDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)
    
    def __getitem__(self, index):
        return self.data[index]['input'], self.data[index]['target']
    
    def __len__(self):
        return self.length
    
train_dataloader = DataLoader(DNDDataset(training_data), batch_size=32, shuffle=True)
test_dataloader = DataLoader(DNDDataset(test_data), batch_size=32, shuffle=True)
valid_dataloader = DataLoader(DNDDataset(valid_data), batch_size=32, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # embedding
        # fc

    def forward(self, x):
        pass

def train():
    pass

def evaluate():
    pass

