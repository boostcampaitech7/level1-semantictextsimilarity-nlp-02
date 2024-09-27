from base.pytorch_base_dataloader import BaseDataset, BaseDataloader
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import warnings
warnings.filterwarnings("ignore")

class CustomDataloader(BaseDataloader) :
    def get_dataloader(self):
        dataset = self.get_dataset()
        if self.dataloader_type == "CNN" :
            # batch is returned as a tuple with dictionary 
            # each dictionary is composed 'input_ids', 'attention_mask'
            # if train mode, last element is composed 'target', 'binary-target'
            collator = SepearteTwoSentenceDatasetCollator(tokenizer=dataset.tokenizer)
        elif self.dataloader_type == "CNNCat" :
            # batch is returned as a tuple with dictionary
            # one is composed 'input_ids', 'attention_mask'
            # another one is target composed 'target', 'binary-target' if train mode
            collator = CNNCatCollator(tokenizer=dataset.tokenizer)
        elif self.dataloader_type == "CLSDiffMul" :
            # batch is returned as a tuple with dictionary
            # one is composed 'input_ids', 'attention_mask', 'token_type_ids'
            # another one is target composed 'target', 'binary-target' if train mode
            collator = SepearteTwoSentenceDatasetCollator(tokenizer=dataset.tokenizer)
        else :
            raise ValueError(f"Unknown dataloader : {self.dataloader_type}")

        return DataLoader(dataset, batch_size = self.batch_size,
                          collate_fn = collator, shuffle = self.train)

    def get_dataset(self):
        if self.dataset_type == "CNN" :
            return SepearteTwoSentenceDataset(self.data_path, 
                              self.pre_trained_model_path,
                              self.train)
        elif self.dataset_type == "CNNCat" :
            return CNNCatDataset(self.data_path, 
                                 self.pre_trained_model_path,
                                 self.train)
        elif self.dataset_type == "CLSDiffMul" : 
            return SepearteTwoSentenceDataset(self.data_path, 
                                 self.pre_trained_model_path,
                                 self.train)
        else :
            raise ValueError(f"Unknown dataset : {self.dataset_type}")


class CNNCatDataset(BaseDataset) :
    def __getitem__(self, index) :
        def tokenize_text(text) : 
            tokens = self.tokenizer(text, 
                                    add_special_tokens = True,
                                    return_tensors = 'pt',
                                    padding = False,
                                    truncation = True)
            return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

        origin_data = self.origin_data.iloc[index,:]
        text_col = ['sentence_1', 'sentence_2']
        text = '[SEP]'.join([origin_data[col] for col in text_col])
        text_input_ids, text_attention_mask = tokenize_text(text)

        inputs = {
            'input_ids': text_input_ids,
            'attention_mask': text_attention_mask
        }

        # If not in test mode, include targets
        if self.train :
            inputs.update({
                'target': origin_data['label'],
                'binary_target': origin_data['binary-label']
            })
        
        return inputs

class CNNCatCollator :
    def __init__(self, tokenizer : PreTrainedTokenizerBase) :
        self.tokenizer = tokenizer
    
    def __call__(self, batch) :
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        inputs = self.tokenizer.pad(
            {'input_ids' : input_ids, 'attention_mask' : attention_mask},
            return_tensors='pt'
        )

        # When adding labels, process based on whether they are tested or not
        targets = None
        if 'target' in batch[0]:
            targets = {
                'target': torch.tensor([item['target'] for item in batch]),
                'binary_target': torch.tensor([item['binary_target'] for item in batch])
            }
        
        return inputs, targets

class SepearteTwoSentenceDataset(BaseDataset) :     
    def __getitem__(self, index) :
        # inputs have 'input_ids', 'token_type_ids', 'attention_mask'
        # if train mode, have 'target', 'binary_target'
        def tokenize_sentence(sentence) :
            tokens = self.tokenizer(sentence, 
                                    return_tensors = 'pt',
                                    padding = False, # Processed by Data Collator.
                                    truncation = True, # Only truncate sentences when necessary.
                                    ) 
            return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

        origin_data = self.origin_data.iloc[index,:]

        # Tokenize both sentences
        sen1, sen1_mask = tokenize_sentence(origin_data['sentence_1'])
        sen2, sen2_mask = tokenize_sentence(origin_data['sentence_2'])
        
        # Combine inputs into one dictionary for both sentences
        inputs = {
            'input_ids_1': sen1,
            'attention_mask_1': sen1_mask,
            'input_ids_2': sen2,
            'attention_mask_2': sen2_mask
        }

        # If not in test mode, include targets
        if self.train :
            inputs.update({
                'targets': origin_data['label'],
                'binary_targets': origin_data['binary-label']
            })

        return inputs

class SepearteTwoSentenceDatasetCollator :
    def __init__(self, tokenizer : PreTrainedTokenizerBase) :
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        # Separate input_ids and attention_mask for each sentence.
        input_ids_1 = [item['input_ids_1'] for item in batch]
        attention_mask_1 = [item['attention_mask_1'] for item in batch]
        input_ids_2 = [item['input_ids_2'] for item in batch]
        attention_mask_2 = [item['attention_mask_2'] for item in batch]

        # Combine the two lists of input_ids and attention_masks
        combined_input_ids = input_ids_1 + input_ids_2
        combined_attention_masks = attention_mask_1 + attention_mask_2

        # Padding both sentence sets in a single pass
        padded_combined = self.tokenizer.pad(
            {'input_ids': combined_input_ids, 
             'attention_mask': combined_attention_masks},
            return_tensors='pt',
            padding=True,
            max_length=None  # Set to None so it uses the longest sequence in combined batch
        )

        # Separate back the padded input_ids and attention_masks
        batch_size = len(input_ids_1)
        padded_input_ids_1 = padded_combined['input_ids'][:batch_size]
        padded_attention_mask_1 = padded_combined['attention_mask'][:batch_size]
        padded_input_ids_2 = padded_combined['input_ids'][batch_size:]
        padded_attention_mask_2 = padded_combined['attention_mask'][batch_size:]

        batch_1 = {'input_ids': padded_input_ids_1,
                   'attention_mask': padded_attention_mask_1}
        batch_2 = {'input_ids': padded_input_ids_2,
                   'attention_mask': padded_attention_mask_2}
                
        # When adding labels, process based on whether they are included
        targets = None
        if 'targets' in batch[0]:
            targets = {
                'targets': torch.tensor([item['targets'] for item in batch]),
                'binary_targets': torch.tensor([item['binary_targets'] for item in batch])
            }

        return batch_1, batch_2, targets