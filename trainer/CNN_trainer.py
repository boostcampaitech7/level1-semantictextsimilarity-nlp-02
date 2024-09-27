import torch
from base.pytorch_base_trainer import BaseTrainer

class CNNTrainer(BaseTrainer) :
    def __init__(self, model, config, log_dir, train_loader, val_loader,
                 optimizer, criterion, metric) :
        super(CNNTrainer, self).__init__(model, config, log_dir, train_loader, val_loader,
                                         optimizer, criterion, metric) 
    
    def run_epoch(self, loader, epoch, train = True) :
        if train : 
            self.model.train()
        else :
            self.model.eval()
        
        total_loss = 0
        total_metric = 0
        with torch.set_grad_enabled(train) : # Backward passes run when 'train = True'
            for batch in loader :
                sen1, sen2, targets = batch
                sen1_input_ids = sen1['input_ids'].long().to(self.device)
                sen2_input_ids = sen2['input_ids'].long().to(self.device)
                targets = targets['target'].to(self.device)

                outputs = self.model(sen1_input_ids, sen2_input_ids)
                
                loss = self.criterion(outputs.float(), targets.float())

                if train :
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_metric += self.metric(outputs, targets).item()
            
        return total_loss / len(loader), total_metric / len(loader)
    
class CNNCatTrainer(BaseTrainer) :
    def __init__(self, model, config, log_dir, train_loader, val_loader,
                 optimizer, criterion, metric) :
        super(CNNCatTrainer, self).__init__(model, config, log_dir, train_loader, val_loader,
                                         optimizer, criterion, metric) 
    
    def run_epoch(self, loader, epoch, train = True) :
        if train : 
            self.model.train()
        else :
            self.model.eval()
        
        total_loss = 0
        total_metric = 0
        with torch.set_grad_enabled(train) : # Backward passes run when 'train = True'
            for batch in loader :
                sen1, sen2, targets = batch
                sen1_input_ids = sen1['input_ids'].long().to(self.device)
                sen1_mask = sen1['attention_mask'].to(self.device)
                sen2_input_ids = sen2['input_ids'].long().to(self.device)
                sen2_mask = sen2['attention_mask'].to(self.device)
                targets = targets['target'].to(self.device)

                outputs = self.model(sen1_input_ids, sen2_input_ids, sen1_mask, sen2_mask)
                
                loss = self.criterion(outputs.float(), targets.float())

                if train :
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_metric += self.metric(outputs, targets).item()
            
        return total_loss / len(loader), total_metric / len(loader)
    
