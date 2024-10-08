from utils.util import load_config
from data_loaders.lightning_loader import Dataloader
import pytorch_lightning as pl
import torch
import pandas as pd

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    config = load_config('config.yaml')

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(config['model_name'], config['batch_size'], config['shuffle'], config['train_path'], 
                            config['dev_path'], config['test_path'], config['predict_path'], config['max_length'])

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=config['max_epoch'], log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load('saves/model.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('output.csv', index=False)
