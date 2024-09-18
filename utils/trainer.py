from base.pytorch_base_trainer import BaseTrainer
from trainer.CNN_trainer import CNNTrainer
from models.CNN import CNNWithEmbedding
## trainer

def get_trainer(trainer_name, config, log_dir, train_loader, val_loader) :
    # return trainer class
    model = get_model(config)
    if trainer_name == "BaseTrainers" :
        return BaseTrainer(model, config, log_dir, train_loader, val_loader)
    elif trainer_name == "CNN" :
        return CNNTrainer(model, config, log_dir, train_loader, val_loader)
    else :
        raise ValueError(f"Unknown trainer : {trainer_name}")

## model

def get_model(config) :
    architecture = config['architecture']
    if architecture == "CNN" :
        return CNNWithEmbedding(config['pre_trained_model_path'],
                                config['num_filters'],
                                config['filter_sizes'],
                                config['fc_sizes'],
                                config['dropout_rate'])
    else :
        raise ValueError(f"Unknown architecture : {architecture}")



    




## Metrics 
# Pearson correlation


