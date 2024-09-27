import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from utils.trainer_utils import TogetherWithDiffTypeSameLengthTrainer
from models.TogetherClassification import TogetherClassicication
from transformers import DebertaV2ForSequenceClassification

class Middleman :
    def __init__(self, config, trainer_name, log_dir,
                 train_loader, val_loader) :
        self.config = config
        self.trainer_name = trainer_name
        self.log_dir = log_dir
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = GetModel(config)
        self.optimizer = Optimizer(config['optimizer'], self.model)
        self.scheduler = Scheduler(self.optimizer, config['scheduler'])
        self.criterion = Criterion(config['criterion'])
        self.metric = Metric(config['metric'])
    
    def get_trainer(self) :
        if self.trainer_name == "TogetherWithDiffTypeSameLengthTrainer" : 
            return TogetherWithDiffTypeSameLengthTrainer(
                 self.config, self.log_dir, 
                 self.train_loader, self.val_loader,
                 self.model, self.optimizer, self.scheduler, 
                 self.criterion, self.metric
            )
        else :
            raise ValueError(f"Unknown trainer : {self.trainer_name}")

class GetModel :
    def __new__(cls, config) :
        model_name = config['model_name']
        if model_name == "TogetherClassicication" :
            return TogetherClassicication(config['pre_trained_path'], config['fc_hidden_sizes'], config['dropout_rate'])
        elif model_name == "DebertaV2ForSequenceClassification" :
            return DebertaV2ForSequenceClassification.from_pretrained(config['pre_trained_path'], 
                                                                      num_labels = 1)
        else :
            raise ValueError(f"Unknown model : {model_name}")

class Optimizer :
    def __new__(cls, optimizer_config, model) :
        optimizer_type = optimizer_config['type']
        optimizer_params = optimizer_config['params']
        if optimizer_type == "Adam" :
            return optim.Adam(model.parameters(), **optimizer_params)
        else :
            raise ValueError(f"Unknown optimizer : {optimizer_type}")

class Scheduler :
    def __new__(cls, optimizer, scheduler_config) :
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config['params']
        if scheduler_type == "StepLR" : 
            return torch.optim.lr_scheduler.StepLR(optimizer, 
                                                   step_size=scheduler_params['step_size'],
                                                   gamma=scheduler_params['gamma'])
        elif scheduler_type == "ExponentialLR" :
            return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                          gamma=scheduler_params['gamma'])
        elif scheduler_type == "Cos" : 
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                              T_max=scheduler_params['T_max'],
                                                              eta_min=scheduler_params['eta_min'])        
        elif scheduler_type == "None" :
            return None
        else :
            raise ValueError(f"Unknown scheduler : {scheduler_type}")

class Criterion :
    def __new__(cls, criterion_name) :
        if criterion_name == "MSE" :
            return nn.MSELoss()
        elif criterion_name == "Pearson" :
            return PearsonLoss()
        elif criterion_name == "MAE" : 
            return nn.L1Loss()
        elif criterion_name == "BCE" :
            return nn.BCEWithLogitsLoss()
        elif criterion_name == "MSE+BCE" :
            return MSE_BCELoss()
        else :
            raise ValueError(f"Unknown criterion : {criterion_name}")

class MSE_BCELoss :
    def __call__(self, preds, targets, binary_targets, pos_weight) :
        preds = preds.reshape(-1)
        binary_preds = (preds > 2).float()

        regression_loss_fn = nn.MSELoss()
        regression_loss = regression_loss_fn(preds, targets)
        
        pos_weight = torch.tensor([pos_weight], device=preds.device)
        binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        binary_loss = binary_loss_fn(binary_preds, binary_targets)
        
        return regression_loss + 0.5 * binary_loss

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

        eps = 1e-8
        corr = covariance / (pred_std * target_std + eps)

        return -corr

class Metric :
    def __new__(cls, metric_name) :
        if metric_name == "Pearson" :
            return PearsonCorrelation()
        else :
            raise ValueError(f"Unknown metric : {metric_name}")

class PearsonCorrelation :
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
        
        eps = 1e-8
        corr = covariance / (pred_std * target_std + eps)

        return corr

