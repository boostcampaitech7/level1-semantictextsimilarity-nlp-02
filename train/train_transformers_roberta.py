import syspath

import random
import pandas as pd

from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch
import torchmetrics
from data_loaders.transformers_bertForRegression_loader import BertDataLoader
from models.bertForRegression_model import RoBERTaForRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.util import load_config
from torch.nn import MSELoss
import torch


# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    config = load_config('configs/config-roberta-l.yaml')

    # 모델 초기화

    dataloader = BertDataLoader(config['model_name'], config['batch_size'], config['shuffle'], config['train_path'],
                                config['dev_path'], config['test_path'], config['predict_path'], config['max_length'])
    dataloader.setup(stage='fit')

    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    model = RoBERTaForRegression.from_pretrained(
        config['model_name']).to(device)

    # 손실 함수와 옵티마이저 설정
    loss_fn = MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'])

    # 예제 학습 과정
    model.train()
    for epoch in range(config['max_epoch']):  # 3 에포크 학습
        model.train()
        loss_step = []
        train_predictions = []
        train_actuals = []
        for batch in tqdm(train_dataloader, desc="Processing", unit="item"):
            # print(batch.keys())
            # print(batch.values())
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # print("labels - ", labels, type(labels), len(labels))

            # if isinstance(labels, list):
            #     labels = torch.tensor(labels).to('gpu')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # print("outputs: ", outputs.squeeze())
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            loss_step.append(loss.item())

            preds = outputs.squeeze().detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            train_predictions.extend(preds)
            train_actuals.extend(labels)
        avg_loss = sum(loss_step) / len(loss_step)
        # 1 epoch 동안의 loss 평균
        print(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.3f}")

        # 검증 데이터셋 평가
        model.eval()
        val_predictions = []
        val_actuals = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                preds = outputs.squeeze().cpu().numpy()  # 예측 값
                labels = labels.cpu().numpy()  # 실제 값

                val_predictions.extend(preds)
                val_actuals.extend(labels)
         # 훈련 데이터셋에 대한 MSE 및 MAE 계산
        train_mse = mean_squared_error(train_actuals, train_predictions)
        train_mae = mean_absolute_error(train_actuals, train_predictions)
        print(f"Train MSE: {train_mse:.3f}")
        print(f"Train MAE: {train_mae:.3f}")
        # 검증 데이터셋에 대한 MSE 및 MAE 계산
        val_mse = mean_squared_error(val_actuals, val_predictions)
        val_mae = mean_absolute_error(val_actuals, val_predictions)
        print(f"Validation MSE: {val_mse:.3f}")
        print(f"Validation MAE: {val_mae:.3f}")

        # 훈련 데이터셋에 대한 피어슨 상관계수 계산
        pearson_corrcoef = torchmetrics.functional.pearson_corrcoef(
            torch.tensor(train_predictions), torch.tensor(train_actuals))
        print("train_pearson", pearson_corrcoef)

        # 검증 데이터셋에 대한 피어슨 상관계수 계산
        pearson_corrcoef = torchmetrics.functional.pearson_corrcoef(
            torch.tensor(val_predictions), torch.tensor(val_actuals))
        print("test_pearson", pearson_corrcoef)

    torch.save(model, 'saves/roberta-model-l.pt')
