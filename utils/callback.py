from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor

def get_checkpoint_callback():
    # 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',  # 체크포인트 저장 경로
        filename='{epoch}-{val_loss:.2f}-{lr:.8f}',  # 파일명 형식
        save_top_k=3,  # 상위 3개의 체크포인트만 저장
        monitor='val_loss',  # 모니터링할 값
        mode='min',  # 'min'이므로 작은 값이 좋을 때 저장
        save_last=True  # 마지막 체크포인트 저장
    )
    return checkpoint_callback

def get_early_stopping_callback():
    # Early stopping 콜백 설정
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,  # val_loss가 5번 연속 개선되지 않으면 중단
        mode="min"
    )
    return early_stopping_callback

def get_lr_monitor_callback():
    # 학습률 모니터링 콜백 설정
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    return lr_monitor