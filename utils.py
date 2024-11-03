import json
import re
import ast
from typing import List, Dict, Optional

def extract_innermost_json(text):
    start_index = -1
    end_index = -1

    # Iterate to find the indices of the innermost {}
    for i, char in enumerate(text):
        if char == '{':
            start_index = i
        elif char == '}':
            end_index = i
            break
    if start_index != -1 and end_index != -1:
        innermost_json = text[start_index:end_index+1]
        return innermost_json
    else:
        return None

def extract_first_json(input_string):
    # Regular expression to find the first {} block
    match = re.search(r'\{.*?\}', input_string)
    if match:
        return match.group(0)
    else:
        return None

def fix_common_json_errors_and_loads(data: str) -> Optional[Dict]:
    
    try:
        return json.loads(data, strict=False)
    except json.decoder.JSONDecodeError as e:
        pass

    data = data.strip().replace('\n', '')
    if data.startswith('```'):
        data = data[3:]
    if data.endswith('```'):
        data = data[:-3]
    if data.startswith('json'):
        data = data[4:]
    
    data = extract_first_json(data)
    
    if not data.endswith('}'):
        data = data+'}'
    
    if not data.startswith('{'):
        data = '{' + data
        data = extract_innermost_json(data)

    try:
        return json.loads(data, strict=False)
    except json.decoder.JSONDecodeError as e:
        pass
    data = data.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    try:
        return json.loads(data, strict=False)
    except json.decoder.JSONDecodeError as e:
        pass

    print('Failed to load json: {}'.format(data))
    return None

def fix_common_json_errors_and_loads_for_revisor(data: str) -> Optional[Dict]:
    
    try:
        return json.loads(data, strict=False)
    except json.decoder.JSONDecodeError as e:
        pass

    data = data.strip().replace('\n', '')
    data = data.replace('"input"', '\"input\"')
    data = data.replace('"positive_document"', '\"positive_document\"')
    data = data.replace("prompt's", "prompt\'s")
    data = data.replace("task's", "prompt\'s")
    if data.startswith('```'):
        data = data[3:]
    if data.endswith('```'):
        data = data[:-3]
    if data.startswith('json'):
        data = data[4:]
    
    data = extract_first_json(data)

    try:
        return json.loads(data, strict=False)
    except json.decoder.JSONDecodeError as e:
        pass

    if not data.endswith('}'+'}'):
        data = data+'}'

    try:
        return ast.literal_eval(data)
    except json.decoder.JSONDecodeError as e:
        pass

    if not data.startswith('{'):
        data = '{' + data
        data = extract_innermost_json(data)

    try:
        return json.loads(data, strict=False)
    except json.decoder.JSONDecodeError as e:
        pass

    data = data.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    try:
        return json.loads(data, strict=False)
    except json.decoder.JSONDecodeError as e:
        pass

    print('Failed to load json: {}'.format(data))
    return None