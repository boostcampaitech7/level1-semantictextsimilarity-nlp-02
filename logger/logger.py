from pytorch_lightning.loggers import TensorBoardLogger

# TensorBoard logger 설정
lr_logger = TensorBoardLogger("tb_logs", name="lr measuring change")