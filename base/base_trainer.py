from abc import ABC, abstractmethod
import os
import torch
from logger.logger import Logger
from logger.visualization import Visualizer

class BaseTrainer(ABC) :
    def __init__(self, config, log_dir, 
                 train_loader, val_loader,
                 model, optimizer, scheduler, criterion, metric) :
        self.config = config
        self.log_dir = log_dir

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stop_patience = config['early_stop_patience']
        self.patience_count = 0

        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.logger = Logger(log_dir)
        self.visualizer = Visualizer()

    def train(self) :
        self.model.to(self.device)
        for epoch in range(self.config['epochs']) :
            train_loss, train_metric = self.run_epoch(self.train_loader, train = True)
            val_loss, val_metric = self.run_epoch(self.val_loader, train = False)

            if self.scheduler is not None :
                self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metric)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metric)
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Train Loss: {train_loss:.4f}")
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Train metric : {train_metric:.4f}")
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Val Loss: {val_loss:.4f}")
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Val metric : {val_metric:.4f}")

            if self.check_and_update(val_loss, epoch) == "stop":
                break
        
        self.save_progress_plot(epoch, True)
    
    @abstractmethod
    def prediction(self, loader) :
        pass

    @abstractmethod
    def run_epoch(self, loader, train) :
        pass

    def check_and_update(self, val_loss, epoch) :
        if val_loss < self.best_val_loss :
            self.best_val_loss = val_loss
            self.save_model()
            self.logger.log(f"Saved best model with Val Loss: {val_loss:.4f}")
            self.patience_counter = 0
        else :
            self.patience_counter += 1
        
        if epoch % self.config['save_intervals'] == 0 :
            self.save_progress_plot(epoch, False)
            
        if self.early_stop_patience <= self.patience_counter :
            self.logger.log(f"Ealry stopping!!\nEpoch [{epoch+1}/{self.config['epochs']}], Val Loss: {val_loss:.4f}")
            return "stop"

        return "continue"
    
    def save_model(self) :
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))

    def save_progress_plot(self, epoch, final) :
        self.visualizer.save_progress_plot(self.train_losses, self.val_losses,
                                           epoch, self.log_dir, 
                                           "loss", final)
        self.visualizer.save_progress_plot(self.train_metrics, self.val_metrics, 
                                           epoch, self.log_dir, 
                                           "metric", final)

