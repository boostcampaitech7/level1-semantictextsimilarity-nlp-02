import torch
from base.pytorch_base_trainer import BaseTrainer


class CNNTrainer(BaseTrainer) :
    def __init__(self, model, config, log_dir, train_loader, val_loader) :
        super(CNNTrainer, self).__init__(model, config, log_dir, train_loader, val_loader) 
    
    def run_epoch(self, loader, epoch, train = True) :
        if train : 
            self.model.train()
        else :
            self.model.eval()
        
        total_loss = 0
        with torch.set_grad_enabled(train) : # Backward passes run when 'train = True'
            for batch in loader :
                sen1, sen2, targets = batch
                sen1, sen2, targets = sen1['input_ids'].long().to(self.device), sen2['input_ids'].long().to(self.device), targets['target'].to(self.device)
                outputs = self.model(sen1, sen2)
                loss = self.criterion(outputs.float(), targets.float())

                if train :
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
        return total_loss / len(loader)