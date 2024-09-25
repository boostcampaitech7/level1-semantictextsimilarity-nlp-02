from transformers import BertModel, BertPreTrainedModel, DebertaV2Model, DebertaV2PreTrainedModel
import torch.nn as nn
import torchmetrics


class BertForRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)  # 단일 실수 값을 출력하도록 설정

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs[1]  # [CLS] 토큰의 임베딩
        regression_output = self.regressor(cls_output)  # 단일 값 예측
        return regression_output


class DeBERTaForRegression(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = DebertaV2Model(config)
        self.regressor = nn.Linear(config.hidden_size, 1)  # 단일 실수 값을 출력하도록 설정

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 임베딩
        regression_output = self.regressor(cls_output)  # 단일 값 예측
        return regression_output
