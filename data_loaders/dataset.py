import torch
from base.base_dataset import BaseDataset

class TogetherWithDiffTypeSameLength(BaseDataset) :
    def __init__(self, data_path, tokenizer, max_length, is_train) :
        super(TogetherWithDiffTypeSameLength, self).__init__(data_path, tokenizer, max_length, is_train)
    def __getitem__(self, idx) :
        encoded = self.tokenizer(
            self.sentence_1[idx],
            self.sentence_2[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        if self.is_train :
            item['targets'] = torch.tensor(self.targets[idx], dtype=torch.float)
            item['binary_targets'] = torch.tensor(self.binary_targets[idx], dtype = torch.float)

        return item
