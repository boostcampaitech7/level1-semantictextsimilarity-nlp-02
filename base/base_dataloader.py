from abc import ABC, abstractmethod

class BaseDataloader(ABC) :
    def __init__(self, dataset_type, batch_size,
                 data_path, tokenizer, max_length, is_train) :
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.is_train = is_train

    @abstractmethod
    def get_dataset(self) :
        pass

    @abstractmethod
    def get_dataloader(self) :
        pass