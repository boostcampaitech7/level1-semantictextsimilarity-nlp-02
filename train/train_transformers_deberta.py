import random
import pandas as pd

from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch
import torchmetrics
from data_loaders.transformers_bertForRegression_loader import BertDataLoader
from models.bertForRegression_model import DeBERTaForRegression
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
    config = load_config('configs/config-deberta.yaml')

    # 모델 초기화

    dataloader = BertDataLoader(config['model_name'], config['batch_size'], config['shuffle'], config['train_path'],
                                config['dev_path'], config['test_path'], config['predict_path'], config['max_length'])
    dataloader.setup(stage='fit')

    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    model = DeBERTaForRegression.from_pretrained(
        config['model_name']).to(device)

    # 손실 함수와 옵티마이저 설정
    loss_fn = MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'])

    # 예제 학습 과정
    model.train()
    print(train_dataloader)
    for epoch in range(config['max_epoch']):  # 3 에포크 학습
        for batch in train_dataloader:
            # print(batch.keys())
            # print(batch.values())
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            print("labels - ", labels, type(labels), len(labels))

            # if isinstance(labels, list):
            #     labels = torch.tensor(labels).to('gpu')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} - Training Loss: {loss.item()}")

    # 검증 데이터셋 평가
    model.eval()
    val_predictions = []
    val_actuals = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.squeeze().cpu().numpy()  # 예측 값
            labels = labels.cpu().numpy()  # 실제 값

            val_predictions.extend(preds)
            val_actuals.extend(labels)

    # 검증 데이터셋에 대한 MSE 및 MAE 계산
    val_mse = mean_squared_error(val_actuals, val_predictions)
    val_mae = mean_absolute_error(val_actuals, val_predictions)
    print(f"Validation MSE: {val_mse:.3f}")
    print(f"Validation MAE: {val_mae:.3f}")

    pearson_corrcoef = torchmetrics.functional.pearson_corrcoef(
        torch.tensor(val_predictions), torch.tensor(val_actuals))
    print("test_pearson", pearson_corrcoef)

    torch.save(model, 'saves/deberta-model.pt')
