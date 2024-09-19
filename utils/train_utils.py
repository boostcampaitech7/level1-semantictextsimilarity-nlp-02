import torch
import torch.nn as nn
import torch.optim as optim
from base.pytorch_base_trainer import BaseTrainer
from trainer.CNN_trainer import CNNTrainer
from models.CNN import CNNWithEmbedding

class Middleman :
    def __init__(self, trainer_name, config, log_dir, 
                 train_loader, val_loader) :
        self.trainer_name = trainer_name
        self.config = config
        self.log_dir = log_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.modeler = GetModel()
        self.model = self.modeler(config)
        self.optimizer = Optimizer(config['optimizer'], self.model)
        self.criterion = Criterion(config['criterion'])
        self.metric = Metric(config['metric'])

    def get_trainer(self) :
        if self.trainer_name == "BaseTrainer" :
            return BaseTrainer(self.model, self.config, self.log_dir,
                               self.train_loader, self.val_loader,
                               self.optimizer, self.criterion, self.metric)
        elif self.trainer_name == "CNNTrainer" :
            return CNNTrainer(self.model, self.config, self.log_dir,
                              self.train_loader, self.val_loader,
                              self.optimizer, self.criterion, self.metric)
        else :
            raise ValueError(f"Unknown trainer : {self.trainer_name}")
    

class GetModel :
    def __call__(self, config) :
        architecture = config['architecture']
        if architecture == "CNN" :
            return CNNWithEmbedding(config['pre_trained_model_path'],
                                    config['num_filters'],
                                    config['filter_sizes'],
                                    config['fc_sizes'],
                                    config['dropout_rate'])
        else :
            raise ValueError(f"Unknown architecture : {architecture}")

## Optimizer
class Optimizer :
    def __new__(cls, optimizer_config, model):
        optimizer_type = optimizer_config['type']
        optimizer_params = optimizer_config['params']
        if optimizer_type == "Adam" :
            return optim.Adam(model.parameters(), **optimizer_params)
        else :
            raise ValueError(f"Unknown optimizer : {optimizer_type}")

## Criterion
class Criterion :
    def __new__(cls, criterion_name) :
        if criterion_name == "MSE" :
            return nn.MSELoss()
        elif criterion_name == "Pearson" :
            return PearsonLoss()
        else :
            raise ValueError(f"Unknown criterion : {criterion_name}")

class PearsonLoss :
    def __call__(self, preds, targets) :
        preds = preds.reshape(-1,1)
        targets = targets.reshape(-1,1)

        pred_mean = preds.mean()
        targets_mean = targets.mean()

        pred_diff = preds - pred_mean
        target_diff = targets - targets_mean

        covariance = (pred_diff * target_diff).mean()
        pred_std = preds.std()
        target_std = targets.std()

        corr = covariance / (pred_std * target_std)

        return -corr

## Metrics 
class Metric :
    def __new__(cls, metric_name) :
        if metric_name == "Pearson" :
            return PearsonCorrelation()
        else :
            raise ValueError(f"Unknown metric : {metric_name}")

class PearsonCorrelation :
    def __call__(self, outputs, targets) :
        outputs = outputs.detach().reshape(1,-1)
        targets = targets.detach().reshape(1,-1)
        cat = torch.cat([outputs, targets])
        return torch.corrcoef(cat)[0,1]

        
