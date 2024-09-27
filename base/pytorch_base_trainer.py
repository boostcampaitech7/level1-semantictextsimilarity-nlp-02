import os
import torch
from logger.logger import Logger
from logger.visualization import Visualizer

class BaseTrainer : 
    def __init__(self, model, config, log_dir, train_loader, val_loader,
                 optimizer, criterion, metric, scheduler = None) :
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping_patience = config['early_stopping_patience']
        self.patience_counter = 0
        self.log_dir = log_dir

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

            if self.scheduler is not None:
                self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metric)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metric)
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Train Loss: {train_loss:.4f}")
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Train metric average : {train_metric:.4f}")
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Val Loss: {val_loss:.4f}")
            self.logger.log(f"Epoch [{epoch+1}/{self.config['epochs']}], Val metric average : {val_metric:.4f}")

            if self.check_and_update(val_loss, epoch) == "stop":
                break

        self.visualizer.save_loss_plot(self.train_losses, self.val_losses, epoch, self.log_dir, final=True)
        self.visualizer.save_metric_plot(self.train_metrics, self.val_metrics, epoch, self.log_dir, final=True)

    def run_epoch(self, loader, train) :
        if train : 
            self.model.train()
        else :
            self.model.eval()
        
        total_loss = 0
        total_metric = 0
        with torch.set_grad_enabled(train) : # Backward passes run when 'train = True'
            for inputs, targets in loader :
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if train : 
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_metric += self.metric(outputs, targets).item()
        
        return total_loss / len(loader), total_metric / len(loader)
    
    def check_and_update(self, val_loss, epoch) :
        if val_loss < self.best_val_loss :
            self.best_val_loss = val_loss
            self.save_model()
            self.logger.log(f"Saved best model with Val Loss: {val_loss:.4f}")
            self.patience_counter = 0
        else :
            self.patience_counter += 1
        
        if epoch % self.config['save_intervals'] == 0 :
                self.visualizer.save_loss_plot(self.train_losses, self.val_losses, epoch, self.log_dir)
                self.visualizer.save_metric_plot(self.train_metrics, self.val_metrics, epoch, self.log_dir)

        if self.early_stopping_patience <= self.patience_counter :
            self.logger.log(f"Ealry stopping!!\nEpoch [{epoch+1}/{self.config['epochs']}], Val Loss: {val_loss:.4f}")
            return "stop"

        return "continue"
    
    def save_model(self) :
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))

