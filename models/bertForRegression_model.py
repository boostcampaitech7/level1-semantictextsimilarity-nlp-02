from transformers import BertModel, BertPreTrainedModel, DebertaV2Model, DebertaV2PreTrainedModel, RobertaModel, RobertaPreTrainedModel
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
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(p=0.5)
        self.regressor = nn.Linear(config.hidden_size, 1)  # 단일 실수 값을 출력하도록 설정

    def forward(self, input_ids, attention_mask=None):
        outputs = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask)
        # cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 임베딩

        # 문장 전체의 모든 토큰 임베딩 사용하여 가중평균 계산

        # attention mask를 outputs.last_hidden_state와 크기를 맞춤
        mask = attention_mask.unsqueeze(-1).expand(
            outputs.last_hidden_state.size()).float()
        # 각 토큰 임베딩에 마스크 적용한 뒤 전체 문장의 총합 임베딩 계산
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
        # 마스크 합을 계산하여 실제로 문장에 해당하는 토큰 수 계산, 최소값 설정하여 0으로 나누는 것 방지
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        # 총합 임베딩을 유효 토큰 수로 나눠서 평균 토큰 임베딩 계산
        cls_output = sum_embeddings / sum_mask

        # Dropout 적용
        cls_output = self.dropout(cls_output)
        regression_output = self.regressor(cls_output)  # 단일 값 예측
        return regression_output


class RoBERTaForRegression(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)  # 단일 실수 값을 출력하도록 설정

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[
            :, 0, :]
        regression_output = self.regressor(cls_output)  # 단일 값 예측
        return regression_output
