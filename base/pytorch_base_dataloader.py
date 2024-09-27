import pandas as pd
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BaseDataset(ABC, Dataset) : 
    def __init__(self, data_path, pre_trained_model_path, train) :
        self.origin_data = pd.read_csv(data_path)
        self.model_path = pre_trained_model_path
        self.train = train
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_path)

    def __len__(self) :
        return self.origin_data.shape[0]
    
    @abstractmethod
    def __getitem__(self, index) :
        pass

class BaseDataloader(ABC) :
    def __init__(self, data_path, dataset_type, dataloader_type,
                 batch_size, 
                 pre_trained_model_path = None, train = True) :
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.dataloader_type = dataloader_type
        self.batch_size = batch_size
        self.pre_trained_model_path = pre_trained_model_path
        self.train = train
    
    @abstractmethod
    def get_dataset(self) :
        pass

    @abstractmethod
    def get_dataloader(self) :
        pass

    



