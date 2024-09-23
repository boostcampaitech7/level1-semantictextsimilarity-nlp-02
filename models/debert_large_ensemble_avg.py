import transformers
import torchmetrics
import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoConfig


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        sd_paths = [
            "saves/models/deberta_large_120-epoch=11-val_pearson=0.934-val_loss=0.396.ckpt",
            "saves/models/deberta_large_100-epoch=14-val_pearson=0.935-val_loss=0.413.ckpt",
            "saves/models/deberta_large_80-epoch=09-val_pearson=0.934-val_loss=0.406.ckpt"
        ]
        sds = []

        for path in sd_paths : 
            temp_ckpt = torch.load(path)
            sds.append(temp_ckpt['state_dict'])

        for key in sds[0]:
            sds[0][key] = torch.stack([sd[key] for sd in sds]).mean(dim=0)
        
        self.plm = torch.load('saves/deberta_large_120.pt')
        self.plm.load_state_dict(sds[0])
        
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer