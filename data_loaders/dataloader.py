from torch.utils.data import DataLoader
from base.base_dataloader import BaseDataloader
from data_loaders.dataset import TogetherWithDiffTypeSameLength

class CustomDataloader :
    def __new__(cls, config, dataloader_type, data_path, tokenizer, is_train) :
        if dataloader_type == "NoCollatorDataloader" :
            loader = NoCollatorDataloader(
                config['dataset_type'], config['batch_size'],
                data_path, tokenizer,
                config['max_length'], is_train)
        else : 
            raise ValueError(f"Unknown dataloader : {dataloader_type}")
        return loader.get_dataloader()

class NoCollatorDataloader(BaseDataloader) :
    def __init__(self, dataset_type, batch_size,
                 data_path, tokenizer, max_length, is_train) :
        super(NoCollatorDataloader, self).__init__(dataset_type, batch_size,
                         data_path, tokenizer, max_length, is_train)
    def get_dataloader(self):
        dataset = self.get_dataset()
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.is_train)

    def get_dataset(self):
        if self.dataset_type == "TogetherWithTypeToken" :
            return TogetherWithDiffTypeSameLength(
                self.data_path, self.tokenizer,
                self.max_length, self.is_train)
        else : 
            raise ValueError(f"Unknown dataset : {self.dataset_type}")