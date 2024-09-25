# baseline model 



### 추가 예정


# bertForRegressionWithTransformersModel

## 개요
대회 데이터셋의 텍스트 - 레이블 간 회귀 관계를 모델링하는 것을 pytorch lightning 대신 hugginface transformers의 BertModel 기반으로 개발
## 환경
> python3, pytorch, transformers
> 

## 파일 구조

### Model
models/bertForRegression_model.py
### DataLoader
data_loaders/transformers_bertForRegression_loader.py
### train
train/train_transformers_regression.py
### test
test/test_transforemers_regression.py
### config
[bert-base-uncased model](configs/config_bert_base-uncased.yaml) - configs/config_bert_base-uncased.yaml


## Installation

1. 타겟 폴더에 sftp 파일 업로드 진행
2. 가상환경 생성


``` sh
 python -m venv 가상환경 네임
```

3. activate.sh 가상환경 path 수정

```sh
. chw_env/bin/activate ## 이것을
. {자신이 만든 가상환경 네임}/bin/activate
```

4. 가상환경 활성화
``` sh
source activate.sh
```

5. 패키지 설치
``` sh
pip install -r requirements.txt
```


## Manual
* 가상환경 활성화 필수


### Train
``` sh
python train/train_transformers_regression.py
```

### Test
``` sh
python test/test_transformers_regression.py
```

### 모델 저장 위치

saves/bert_base_advanced_model.pt


## Notice
1. 로그 출력

    tqdm 사용 예정

2. 모델 체크 포인트 추가 예정

3. train 스크립트와 test 스크립트에 있는 훈련 과정을 models 폴더 아래에 있는 파일에 캡슐화 예정

4. 에포크 혹은 스텝마다 측정된 지표들(mse, mae) 등등을 그래프화 예정

5. optuna 혹은 pytorch scheduler를 이용한 하이퍼파라미터 튜닝 자동화 추가 예정 
