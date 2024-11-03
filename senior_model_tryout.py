import torch
import os
import random
import numpy as np
import json
import re

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts_synthesis import get_create_classify_data_prompt
from utils import fix_common_json_errors_and_loads

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LLAMA3_PROMPT = """
{prompt} [/INST]
""".strip("\n")

# Each query must come with a one-sentence instruction that describes the task
tasks = [
    'Identify the intended age group for educational technology products.',
    'Classify businesses based on their operational hours.'
]
language = 'English'

prompts = [LLAMA3_PROMPT.format(prompt=get_create_classify_data_prompt(task=task, language=language)[1]['content']) for task in tasks]

tokenizer = AutoTokenizer.from_pretrained('Haon-Chen/speed-synthesis-7b-senior')
model = AutoModelForCausalLM.from_pretrained('Haon-Chen/speed-synthesis-7b-senior')
model.to("cuda:0")
model.eval()
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

with torch.inference_mode():
    # Tokenize the input texts
    encodes = tokenizer(prompts, padding="longest", add_special_tokens=True, return_tensors="pt")
    input_ids = encodes.input_ids.to(model.device)
    attention_mask = encodes.attention_mask.to(model.device)

    # Set the generation parameters
    GEN_CONFIG = {"do_sample":True, "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 800}
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id = tokenizer.eos_token_id,
        **GEN_CONFIG
    )
output_texts = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
batch_results = []
for i in range(len(output_texts)):
    batch_results.append(output_texts[i][len(prompts[i]):].strip(' '))

# Format outputs
bad_cnt=0
outputs = []
for i, result in enumerate(batch_results):
    try:
        output = fix_common_json_errors_and_loads(result)
        user_query = output.get("input_text", "")
        positive_document = output.get("label", "")
        hard_negative_document = output.get("misleading_label", "")
    except:
        bad_cnt+=1
        continue
    out_data = {
        "query": user_query,
        "positives": [positive_document],
        "negatives": [hard_negative_document],
        "language": "English",
        "task_definition": tasks[i],
    }
    outputs.append(out_data)
print(bad_cnt)
print(outputs)