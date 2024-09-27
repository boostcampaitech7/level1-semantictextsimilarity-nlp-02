import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from utils.utils import load_config
from utils.train_utils import Middleman, GetModel
from data_loaders.dataloader import CustomDataloader

if __name__ == "__main__" : 
    config_path = sys.argv[1]
    config = load_config(config_path)

    # setting log directory
    directory_name = os.path.splitext(os.path.basename(config_path))[0]
    log_dir = f'saved/{directory_name}'
    os.makedirs(log_dir, exist_ok = True)
    # save config file in log directory
    shutil.copy(config_path, os.path.join(log_dir, 'config.yaml'))
    
    tokenizer = AutoTokenizer.from_pretrained(config['pre_trained_path'])
    # get dataloader
    train_loader = CustomDataloader(config, config['dataloader_type'], 
                                    config['train_path'], tokenizer, True)
    val_loader = CustomDataloader(config, config['dataloader_type'],
                                  config['val_path'], tokenizer, True)
    
    middleman = Middleman(config, config['trainer_name'], log_dir, 
                          train_loader, val_loader)
    trainer = middleman.get_trainer()
    trainer.train()

    # setting test log directory
    test_log_dir = f'saved/{directory_name}/test'
    os.makedirs(test_log_dir, exist_ok = True)

    # load best model
    model_path = f'saved/{directory_name}/best_model.pth'
    state_dict = torch.load(model_path)

    # test loader
    test_loader = CustomDataloader(config, config['dataloader_type'],
                                   config['test_path'], tokenizer, False)
    results = trainer.prediction(state_dict, test_loader) 
    results = np.round(np.array(results), 1)

    output = pd.read_csv(config['submission_path'])
    output['target'] = results
    output.to_csv(os.path.join(test_log_dir, "output.csv"), index = False)

    

    