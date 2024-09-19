import os
import pandas as pd
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class SemanticTextSimilarityDataset(Dataset) : 
    def __init__(self, data_path, pre_trained_model_path, train) :
        self.origin_data = pd.read_csv(data_path)
        self.model_path = pre_trained_model_path
        self.train = train
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_path,
                                                       trust_remote_code = True)

    def __len__(self) :
        return self.origin_data.shape[0]
    
    @abstractmethod
    def __getitem__(self, index) :
        pass



