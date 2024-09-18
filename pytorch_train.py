import os
import sys
import shutil
from utils.util import load_config
from utils.trainer import get_trainer
from data_loaders.pytorch_loader import get_dataloader


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
    train_loader = get_dataloader(config['train_path'], config, train = True)
    val_loader = get_dataloader(config['val_path'], config, train = True)

    trainer = get_trainer(config['trainer'], config, log_dir, train_loader, val_loader)
    trainer.train()