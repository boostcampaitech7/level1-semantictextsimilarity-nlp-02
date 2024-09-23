import random
import pandas as pd

from tqdm.auto import tqdm

import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.lightning_loader import Dataloader
from models.lightning_model_L2loss import Model
from utils.util import load_config
import os

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    config = load_config('configs/deberta_large_100.yaml')


    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(config['model_name'], config['batch_size'], config['shuffle'], config['train_path'], 
                            config['dev_path'], config['test_path'], config['predict_path'], config['max_length'], num_workers = 7)
    model = Model(config['model_name'], config['learning_rate'])
    
    
    # checkpoint settings
    checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    monitor="val_pearson",
    mode="max",
    dirpath="saves/models/",
    filename=config['model_nick']+"-L2-{epoch:02d}-{val_pearson:.3f}-{val_loss:.3f}",
    )

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=config['max_epoch'], log_every_n_steps=1, callbacks = [checkpoint_callback])

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'saves/'+config['model_nick']+'-L2.pt')