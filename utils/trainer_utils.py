import torch
from base.base_trainer import BaseTrainer

class TogetherWithDiffTypeSameLengthTrainer(BaseTrainer) :
    def __init__(self, config, log_dir, 
                 train_loader, val_loader,
                 model, optimizer, scheduler, criterion, metric) :
        super(TogetherWithDiffTypeSameLengthTrainer, self).__init__(config, log_dir, 
                                                                    train_loader, val_loader,
                                                                    model, optimizer, scheduler, criterion, metric)
    
    def run_epoch(self, loader, train) :
        if train : 
            self.model.train()
        else :
            self.model.eval()
        
        total_loss = 0
        total_metric = 0
        with torch.set_grad_enabled(train) :
            for batch in loader :
                input_ids = batch['input_ids'].long().to(self.device)
                attention_mask = batch['attention_mask'].long().to(self.device)
                targets = batch['targets'].float().to(self.device)
                binary_targets = batch['binary_targets'].float().to(self.device)

                outputs = self.model(input_ids, attention_mask)

                loss = self.criterion(outputs, targets, binary_targets, self.config['pos_weight'])

                if train :
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_metric += self.metric(outputs, targets).item()

        return total_loss / len(loader), total_metric / len(loader)

    def prediction(self, state_dict, loader) : 
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        preds = []
        with torch.no_grad() : 
            for batch in loader :
                input_ids = batch['input_ids'].long().to(self.device)
                attention_mask = batch['attention_mask'].long().to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds.extend(outputs.cpu().numpy().flatten())
        return preds
        
