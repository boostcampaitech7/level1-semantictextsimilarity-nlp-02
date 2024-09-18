import os
import sys
import pandas as pd 
import numpy as np
import torch 
from utils.utils import load_config
from utils.train_utils import GetModel
from utils.test_utils import Tester
from data_loaders.pytorch_loader import CustomDataloader


if __name__ == "__main__" :
    config_path = sys.argv[1]
    config = load_config(config_path)

    # setting log directory
    directory_name = os.path.splitext(os.path.basename(config_path))[0]
    log_dir = f'saved/{directory_name}/test'
    os.makedirs(log_dir, exist_ok = True)

    # model 
    model_path = f'saved/{directory_name}/best_model.pth'
    state_dict = torch.load(model_path)
    modeler = GetModel()
    model = modeler(config)
    model.load_state_dict(state_dict)

    # get dataloader
    test = CustomDataloader(config['test_path'], config['dataset_type'], config['dataloader_type'],
                            config['batch_size'], config['pre_trained_model_path'],train=False)
    test_loader = test.get_dataloader()

    tester = Tester(config, model, config['trainer'], test_loader,
                    log_dir, config['attention_based'])
    
    results = tester.test()
    results = np.round(np.array(results), 1)

    output = pd.read_csv(config['submission_path'])
    output['target'] = results
    output.to_csv(os.path.join(log_dir, "output.csv"), index = False)
