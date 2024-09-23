import pandas as pd
import os

from tqdm import tqdm
import transformers
import torch
######## 본인 huggingface Access Token 입력 아마 안넣어도 상관없을듯? original llama 실헝때 사용했음. ######
#os.environ['HF_TOKEN'] = '' 
########################################################################################################

#sent1, sent2 laod
sent1 = pd.read_csv("../data/sent1.csv", header = None).iloc[:, 1]
sent2 = pd.read_csv("../data/sent2.csv", header = None).iloc[:, 1]

# 한국어 기반으로 LLama3 fine-tuning한 모델
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# model의 입력 형식을 지정해 주는 부분 (수정할 필요 없음)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# model을 추론전용으로 전환
pipeline.model.eval()

######## 이부분이 prompt로, 이부분을 변경하면서 llm에게 동작을 명령할 수 있음 ###############
PROMPT = '''Translate following korean sentence to English sentence. Don't use quotation marks.'''
########################################################################################

######### Sent1을 입력으로 받아 prompt에서 지정된 명령을 수행한 후 new_sent1.csv를 저장하는 부분#######
tl_sent1 = []
for sentence in tqdm(sent1) :
    instruction = sentence

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    tl_sent1.append(outputs[0]["generated_text"][len(prompt):])

pd.DataFrame(tl_sent1).to_csv("../data/augmented/tl_sent1.csv",header = False )


######### Sent2을 입력으로 받아 prompt에서 지정된 명령을 수행한 후 new_sent2.csv를 저장하는 부분#######
tl_sent2 = []
for sentence in tqdm(sent2) :
    instruction = sentence

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    tl_sent2.append(outputs[0]["generated_text"][len(prompt):])

pd.DataFrame(tl_sent2).to_csv("../data/augmented/tl_sent2.csv",header = False)