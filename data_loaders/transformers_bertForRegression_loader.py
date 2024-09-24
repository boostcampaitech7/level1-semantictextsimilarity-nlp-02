import torch
import transformers
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, DataCollatorWithPadding
import pytorch_lightning as pl
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class RegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 텍스트 토크나이징
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)  # [1, seq_len] -> [seq_len]
        attention_mask = inputs['attention_mask'].squeeze(
            0)  # [1, seq_len] -> [seq_len]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float)
        }


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        print(f"len(texts) = {len(texts)}, len(labels) = {len(labels)}")
        # assert len(texts) == len(labels), "texts와 labels의 길이가 다릅니다."

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx >= len(self.texts):
            raise IndexError(
                f"Index {idx} is out of range for texts list with length {len(self.texts)}")

        text = self.texts[idx]

        # 토크나이저를 사용하여 input_ids와 attention_mask 생성
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 반환할 때 딕셔너리 형태로 반환
        returnOutput = {
            # (1, max_length) -> (max_length,)
            'input_ids': encoding['input_ids'].squeeze(),
            # (1, max_length) -> (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        if len(self.labels) != 0:
            returnOutput['labels'] = self.labels[idx]

        return returnOutput


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


class BertDataLoader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, max_length):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_length)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.text_columns = ['sentence_1', 'sentence_2']
        self.delete_columns = ['id']
        self.target_columns = ['label']

    def tokenizing(self, dataframe):
        texts = []
        for _, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column]
                                for text_column in self.text_columns])
            texts.append(text)
        return texts
    # def tokenizing(self, dataframe):
    #     data = []
    #     for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
    #         # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
    #         text = '[SEP]'.join([item[text_column]
    #                             for text_column in self.text_columns])
    #         outputs = self.tokenizer(
    #             text, add_special_tokens=True, padding='max_length', truncation=True)
    #         data.append(outputs['input_ids'])
    #     return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except Exception as e:
            print(e)
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)
        print('preprocessing targets', targets)
        return inputs, targets

    def load_dataset(self, path):
        # CSV 파일을 불러와서 텍스트와 레이블 분리
        data = pd.read_csv(path)
        texts = self.tokenizing(data)
        try:
            targets = data[self.target_column].tolist()
        except KeyError:
            targets = []
        return texts, targets

    def setup(self, stage=None):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = TextDataset(
                train_inputs, train_targets, self.tokenizer)
            self.val_dataset = TextDataset(
                val_inputs, val_targets, self.tokenizer)

        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = TextDataset(
                test_inputs, test_targets, self.tokenizer)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = TextDataset(
                predict_inputs, [], self.tokenizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
