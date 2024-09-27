import os
import sys
import shutil
from utils.utils import load_config
from utils.train_utils import Middleman
from data_loaders.pytorch_loader import CustomDataloader

if __name__ == "__main__" : 
    config_path = sys.argv[1]
    config = load_config(config_path)

    # setting log directory
    directory_name = os.path.splitext(os.path.basename(config_path))[0]
    log_dir = f'saved/{directory_name}'
    os.makedirs(log_dir, exist_ok = True)
    # save config file in log directory
    shutil.copy(config_path, os.path.join(log_dir, 'config.yaml'))
    
    # get dataloader
    train = CustomDataloader(config['train_path'], config['dataset_type'], config['dataloader_type'],
                                    config['batch_size'], config['pre_trained_model_path'],train=True)
    val = CustomDataloader(config['val_path'], config['dataset_type'], config['dataloader_type'],
                                    config['batch_size'], config['pre_trained_model_path'],train=True)
    
    train_loader = train.get_dataloader()
    val_loader = val.get_dataloader()
    
    # get trainer and start train
    middleman = Middleman(config['trainer'], config, log_dir,
                          train_loader, val_loader)
    trainer = middleman.get_trainer()
    trainer.train()