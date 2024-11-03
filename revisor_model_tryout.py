import torch
import os
import random
import numpy as np
import json


from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional

from prompts_aligning import get_create_all_revise_data_prompt
from utils import fix_common_json_errors_and_loads_for_revisor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LLAMA3_PROMPT = """
{prompt} [/INST]
""".strip("\n")

# Each query must come with a one-sentence instruction that describes the task
old_prompts = [
    "You have been assigned a text matching task: Match a Stockard Channing movie title with a brief plot description.\n\nYour mission is to write one example for this task in JSON format. The JSON object must contain the following keys:\n- \"input\": a string, a random input specified by the task.\n- \"positive_document\": a string, a relevant document for the \"input\" according to the task.\n\nPlease adhere to the following guidelines:\n- The values of all fields should be in English.\n- Both the \"input\" and \"positive_document\" should be very short (a sentence or a phrase), avoid substantial word overlaps, otherwise the task would be too easy.\n- The \"input\" and \"positive_document\" should be independent of each other.\n\nYour output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"
]
old_data = [
    {"input": "Stockard Channing in 'The Business of Strangers', directed by Patrick Stettner.", "positive_document": "In 'The Business of Strangers', Channing stars as a businesswoman who embarks on a ruthless journey, after which she undergoes a drastic change. She faces many challenges while pursuing her goals and eventually comes out stronger."},
]
language = 'English'

prompts = [LLAMA3_PROMPT.format(prompt=get_create_all_revise_data_prompt(prompt=old_prompt, data=json.dumps(data))[1]['content']) for old_prompt in old_prompts for data in old_data]

tokenizer = AutoTokenizer.from_pretrained('Haon-Chen/speed-synthesis-7b-revisor')
model = AutoModelForCausalLM.from_pretrained('Haon-Chen/speed-synthesis-7b-revisor')
model.to("cuda:0")
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# Tokenize the input texts
encodes = tokenizer(prompts, padding="longest", add_special_tokens=True, return_tensors="pt")
input_ids = encodes.input_ids.to(model.device)
attention_mask = encodes.attention_mask.to(model.device)

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

bad_cnt=0
outputs = []
for i, result in enumerate(batch_results):
    try:
        content = fix_common_json_errors_and_loads_for_revisor(result)
        revision = content["revision"]
        reason = content["reason"]

        user_query = revision.get("input", "")
        positive_document = revision.get("positive_document", "")
    except:
        bad_cnt+=1
        continue
    out_data = {
        "query": user_query,
        "positives": [positive_document],
        "negatives": [],
        "language": "English",
        "reason": reason,
    }
    outputs.append(out_data)
print(bad_cnt)
print(outputs)