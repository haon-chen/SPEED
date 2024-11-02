import random
import numpy as np

from typing import List, Dict, Optional




def get_create_retrieval_data_prompt(task: str, language: Optional[str] = None) -> List[Dict]:
    if language is None:
        language = 'English'
    num_words: int = random.choice([50, 100, 200, 300, 400, 500])
    difficulty: str = random.choice(['high school', 'college', 'PhD'])
    query_length: str = random.choices(
        ['less than 5 words', '5 to 15 words', 'at least 10 words'],
        weights=[0.4, 0.4, 0.2]
    )[0]
    clarity: str = random.choices(
        ['clear', 'understandable with some effort', 'ambiguous'],
        weights=[0.6, 0.2, 0.2]
    )[0]
    query_type: str = random.choices(
        ['extremely long-tail', 'long-tail', 'common'],
        weights=[0.4, 0.5, 0.1]
    )[0]

    guidelines: List[str] = [
        f'- The "user_query" should be {query_type}, {query_length}, {clarity}, and diverse in topic.',
        '- All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable if some parts of the "positive_document" are not topically related to the query.',
        f'- All documents should be at least {num_words} words long.',
        '- The "hard_negative_document" contains some useful information, but it should be less useful or comprehensive compared to the "positive_document".',
        f'- Both the query and documents should be in {language}.',
        '- Do not provide any explanation in any document on why it is relevant or not relevant to the query.',
    ]
    if random.random() < 0.5:
        guidelines.append(f'- Both the query and documents require {difficulty} level education to understand.')
    np.random.shuffle(guidelines)

    str_guideline: str = '\n'.join(guidelines)

    prompt = f"""You have been assigned a retrieval task: {task}
The \"user_query\" should be
Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following keys:
- "user_query": a string, a random user search query specified by the retrieval task.
- "positive_document": a string, a relevant document for the user query.
- "hard_negative_document": a string, a hard negative document that only appears relevant to the query.

Please adhere to the following guidelines:
{str_guideline}

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_sts_prompt_with_topic(topic, language: Optional[str] = None) -> List[Dict]:
    if language is None:
        language = 'English'

    high_score: float = random.choice([4, 4.5, 5])
    low_score: float = random.choice([2.5, 3, 3.5])
    unit: str = random.choices(
        ['sentence', 'phrase', 'passage'],
        weights=[0.9, 0.05, 0.05], k=1
    )[0]
    difficulty: str = random.choice(['elementary school', 'high school', 'college'])

    guidelines: List[str] = [
        f'- The keys in JSON are "S1", "S2", and "S3", the values are all strings in {language}, do not add any other keys.',
        f'- There should be some word overlaps between all three {unit}s.',
        f'- The similarity score between S1 and S2 should be {high_score}.',
        f'- The similarity score between S1 and S3 should be {low_score}.',
    ]
    if random.random() < 0.5:
        guidelines.append(f'- The {unit}s require {difficulty} level education to understand and should be diverse in terms of topic and length.')

    str_guideline: str = '\n'.join(guidelines)

    prompt: str = f"""Write a {unit} triple for the topic: {topic} with varying semantic similarity scores in JSON format. The semantic similarity score ranges from 1 to 5, with 1 denotes least similar and 5 denotes most similar.

Please adhere to the following guidelines:
{str_guideline}

Your output must always be a JSON object only with three keys "S1", "S2" and "S3", do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages



def get_create_classify_data_prompt(task: str, language: Optional[str] = None) -> List[Dict]:
    if language is None:
        language = 'English'
    num_words: str = random.choices(
        ["less than 10", "at least 10", "at least 50", "at least 100", "at least 200"],
        weights=[0.1, 0.3, 0.2, 0.2, 0.2]
    )[0]
    difficulty: str = random.choice(['high school', 'college', 'PhD'])
    clarity: str = random.choices(
        ['clear', 'understandable with some effort', 'ambiguous'],
        weights=[0.6, 0.2, 0.2]
    )[0]

    guidelines: List[str] = [
        f'- The "input_text" should be {num_words} words and diverse in expression.',
        '- The "misleading_label" must be a valid label for the given task, but not as appropriate as the "label" for the "input_text".',
        f'- The values for all fields should be in {language}.',
        '- Avoid including the values of the "label" and "misleading_label" fields in the "input_text", that would make the task too easy.',
    ]
    if random.random() < 0.5:
        guidelines.append(f'- The "input_text" is {clarity} and requires {difficulty} level education to comprehend.')
    np.random.shuffle(guidelines)

    str_guideline: str = '\n'.join(guidelines)

    prompt = f"""You have been assigned a text classification task: {task}

Your mission is to write one text classification example for this task in JSON format. The JSON object must contain the following keys:
- "input_text": a string, the input text specified by the classification task.
- "label": a string, the correct label of the input text.
- "misleading_label": a string, an incorrect label that is related to the task.  

Please adhere to the following guidelines:
{str_guideline}

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_create_s2s_p2p_data_prompt(task: str, task_type: str, language: Optional[str] = None) -> List[Dict]:
    assert task_type in ['s2s', 'p2p']
    if language is None:
        language = 'English'

    if task_type == 's2s':
        length_requirement: str = "very short (a sentence or a phrase)"
    else:
        length_requirement: str = "long documents (at least 300 words)"

    guidelines: List[str] = [
        f'- The values of all fields should be in {language}.',
        f'- Both the "input" and "positive_document" should be {length_requirement}, avoid substantial word overlaps, otherwise the task would be too easy.',
        '- The "input" and "positive_document" should be independent of each other.',
    ]
    str_guideline: str = '\n'.join(guidelines)

    prompt = f"""You have been assigned a text matching task: {task}

Your mission is to write one example for this task in JSON format. The JSON object must contain the following keys:
- "input": a string, a random input specified by the task.
- "positive_document": a string, a relevant document for the "input" according to the task.

Please adhere to the following guidelines:
{str_guideline}

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages

