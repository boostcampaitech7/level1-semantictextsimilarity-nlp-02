from base.pytorch_base_dataloader import SemanticTextSimilarityDataset
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import warnings

warnings.filterwarnings("ignore")


def get_dataloader(data_path, config, train = True) :
    dataloader_type = config['dataloader']
    if dataloader_type == "CNN" :
        return get_CNNDataloader(data_path, config['pre_trained_model_path'], config['batch_size'], train = train)
    else :
        raise ValueError(f"Unknown dataloader : {dataloader_type}")

class CNNDataset(SemanticTextSimilarityDataset) :     
    def __getitem__(self, index) :
        origin_data = self.origin_data.iloc[index,:]

        def tokenize_sentence(sentence) :
            tokens = self.tokenizer(sentence, 
                                    return_tensors = 'pt',
                                    padding = False, # Processed by Data Collator.
                                    truncation = True, # Only truncate sentences when necessary.
                                    ) 
            return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

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
                'target': origin_data['label'],
                'binary_target': origin_data['binary-label']
            })

        return inputs

class CNNCollator :
    def __init__(self, tokenizer : PreTrainedTokenizerBase) :
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        # Separate input_ids and attention_mask for each sentence.
        input_ids_1 = [item['input_ids_1'] for item in batch]
        attention_mask_1 = [item['attention_mask_1'] for item in batch]
        input_ids_2 = [item['input_ids_2'] for item in batch]
        attention_mask_2 = [item['attention_mask_2'] for item in batch]

        # Padding for the first sentence
        batch_1 = self.tokenizer.pad(
            {'input_ids': input_ids_1, 'attention_mask': attention_mask_1},
            return_tensors='pt'
        )

        # Padding for the second sentence
        batch_2 = self.tokenizer.pad(
            {'input_ids': input_ids_2, 'attention_mask': attention_mask_2},
            return_tensors='pt'
        )

        # When adding labels, process based on whether they are tested or not
        targets = None
        if 'target' in batch[0]:
            targets = {
                'target': torch.tensor([item['target'] for item in batch]),
                'binary_target': torch.tensor([item['binary_target'] for item in batch])
            }

        return batch_1, batch_2, targets

def get_CNNDataloader(data_path, pre_trained_model_path, batch_size, train = True) : 
    dataset = CNNDataset(data_path = data_path, 
                         pre_trained_model_path = pre_trained_model_path,
                         train = train)
    collator = CNNCollator(tokenizer = dataset.tokenizer)
    dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn = collator, shuffle = train)

    return dataloader
    # batch is returned as a tuple with dictionary 
    # each dictionary is composed 'input_ids', 'attention_mask'
    # if train mode, last element is composed 'target', 'binary-target'