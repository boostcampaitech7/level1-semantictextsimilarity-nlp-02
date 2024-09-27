from abc import ABC, abstractmethod
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset) :
    def __init__(self, data_path, tokenizer, max_length, is_train) :
        self.is_train = is_train
        self.data = pd.read_csv(data_path)
        self.sentence_1 = self.data['sentence_1'].tolist()
        self.sentence_2 = self.data['sentence_2'].tolist()
        if self.is_train :
            self.targets = self.data['label'].tolist()
            self.binary_targets = self.data['binary-label']
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) :
        return len(self.sentence_1)
    
    @abstractmethod
    def __getitem__(self, idx) :
        pass