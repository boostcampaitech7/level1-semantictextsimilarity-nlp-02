import torch
from logger.visualization import Visualizer

class Tester :
    def __init__(self, config, model, trainer_name, test_loader, log_dir, attention_based = False) :
        # trainer_name : To use the same format as when training
        self.tester = self.get_tester(trainer_name)
        self.model = model
        self.loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # For visualization 
        self.config = config # Not sure to use 
        self.log_dir = log_dir
        self.attention_based = attention_based
        self.visualizer = Visualizer()

    def test(self) :
        preds = self.tester(self.model, self.loader, self.device)
        return preds

    def get_tester(self, trainer_name) :
        if trainer_name == "CNNTrainer" :
            return CNNTester()
        elif trainer_name == "CLSDiffMulTrainer" :
            return CLSDiffMulTrainer()

class CLSDiffMulTrainer :
    def __call__(self, model, loader, device) :
        model.to(device)
        model.eval()
        preds = []
        with torch.no_grad() :
            for batch in loader :
                sen1, sen2, _ = batch
                sen1_input_ids = sen1['input_ids'].long().to(device)
                sen1_mask = sen1['attention_mask'].to(device)
                sen2_input_ids = sen2['input_ids'].long().to(device)
                sen2_mask = sen2['attention_mask'].to(device)

                outputs = model(sen1_input_ids, sen1_mask, sen2_input_ids, sen2_mask)
                preds.extend(outputs.cpu().numpy().flatten())
        return preds

class CNNTester :
    def __call__(self, model, loader, device) :
        model.to(device)
        model.eval()
        preds = []
        with torch.no_grad() :
            for batch in loader :
                sen1, sen2, _ = batch
                sen1, sen2 = sen1['input_ids'].long().to(device), sen2['input_ids'].long().to(device)
                outputs = model(sen1, sen2)
                preds.extend(outputs.cpu().numpy().flatten())
        
        return preds
