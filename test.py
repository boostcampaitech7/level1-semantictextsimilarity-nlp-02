from utils.util import load_config
from data_loaders.lightning_loader import Dataloader
from models.lightning_model_L2loss import Model

import pytorch_lightning as pl
import torch
import pandas as pd

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    config = load_config('configs/deberta_large_120.yaml')

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(config['model_name'], config['batch_size'], config['shuffle'], config['train_path'], 
                            config['dev_path'], config['test_path'], config['predict_path'], config['max_length'], num_workers = 7)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=config['max_epoch'], log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load('saves/'+config['model_nick']+'-L2.pt')
    checkpoint = torch.load('saves/models/deberta_large_120-L2-epoch=11-val_pearson=0.939-val_loss=0.264.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv(config['model_nick']+'_L2-output.csv', index=False)
