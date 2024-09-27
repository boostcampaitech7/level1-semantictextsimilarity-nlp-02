import syspath
from utils.util import load_config
from data_loaders.transformers_bertForRegression_loader import BertDataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytorch_lightning as pl
import torch
import pandas as pd
import torchmetrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    config = load_config('configs/config-deberta.yaml')

    # dataloader와 model을 생성합니다.
    dataloader = BertDataLoader(config['model_name'], config['batch_size'], config['shuffle'], config['train_path'],
                                config['dev_path'], config['test_path'], config['predict_path'], config['max_length'])
    dataloader.setup()

    test_dataloader = dataloader.test_dataloader()
    predict_dataloader = dataloader.predict_dataloader()
    # # 토크나이저 로드
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # 평가 데이터셋 인스턴스 생성
    # eval_dataset = RegressionDataset(
    #     texts=eval_df['text'].tolist(),
    #     labels=eval_df['label'].tolist(),
    #     tokenizer=tokenizer
    # )

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load(
        'saves/deberta-model-xlarge.pt').to(device)

    # predictions = trainer.predict(model=model, datamodule=dataloader)

    test_predictions = []
    test_actuals = []
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.squeeze().cpu().numpy()  # 예측 값
            labels = labels.cpu().numpy()  # 실제 값

            test_predictions.extend(preds)
            test_actuals.extend(labels)

        for batch in predict_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.squeeze().cpu().numpy()  # 예측 값

            predictions.extend(preds)

    # 테스트 데이터셋에 대한 MSE 및 MAE 계산
    test_mse = mean_squared_error(test_actuals, test_predictions)
    test_mae = mean_absolute_error(test_actuals, test_predictions)
    print(f"Test MSE: {test_mse:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    pearson_corrcoef = torchmetrics.functional.pearson_corrcoef(
        torch.tensor(test_predictions), torch.tensor(test_actuals))
    print("test_pearson", pearson_corrcoef)
    test_predictions = list(round(float(i), 1) for i in test_predictions)
    print(test_predictions)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.

    predictions = list(round(float(i), 1) for i in predictions)
    print(predictions)
    print(len(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('deberta_output.csv', index=False)
