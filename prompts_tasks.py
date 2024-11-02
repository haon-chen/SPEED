import random
import numpy as np

from typing import List, Dict, Optional



def get_brainstorm_retrieval_tasks_prompt_with_topic_ICL(topic) -> List[Dict]:

    example_tasks: List[str] = [
        "- Retrieve relevant documents for a short keyword web search query that asks for weather information.",
        "- Search for documents that answers a FAQ-style query on children's nutrition.",
        "- Provided a scientific claim as query, retrieve documents that help verify or refute the claim.",
        "- Given a debate topic as query, retrieve documents that is either supportive or against the topic.",
        "- The query is a multi-hop question on celebrities, retrieve documents that help answer the question.",
        "- The query is a natural language description of a function, find the correct code implementation.",
        "- Given a news article that user has read as the query, retrieve other articles that the user may be interested in.",
    ]

    # create diversity for prompts
    np.random.shuffle(example_tasks)
    str_tasks: str = '\n'.join(example_tasks)

    prompt = f"""Brainstorm a list of potentially useful text retrieval tasks for the topic: {topic}.

Here are a few examples for your reference:
{str_tasks}

Please adhere to the following guidelines:
- Specify what the query is, and what the desired documents are.
- Each retrieval task should cover a wide range of queries, and should not be too specific.

Your output must always be a python list of strings only, with about 5 elements, and each element corresponds to a distinct retrieval task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_brainstorm_classify_tasks_prompt_with_topic(topic) -> List[Dict]:
    
    example_tasks: List[str] = [
        "- Classify the given TripAdvisor review into positive, negative, or neutral sentiment.",
        "- Determining if a message is a question, request, command, or statement.",
        "- Classify the given forum post into one of the categories: question, answer, or discussion.",
        "- Given an Amazon product review, find the main features mentioned.",
        "- Given a user comment on a news article, find the main argument presented.",
        "- Categorizing online articles into user-preferred reading difficulty levels.",
        "- Classify the given Yelp review as either helpful or not helpful.",
    ]

    np.random.shuffle(example_tasks)
    str_tasks: str = '\n'.join(example_tasks)
    
    prompt = f"""Brainstorm a list of potentially useful text classification tasks for the topic: {topic}.

Here are a few examples for your reference:
{str_tasks}

Please adhere to the following guidelines:
- Tasks should cover a diverse range of domains and task types.
- Avoid generate tasks similar to classification of sentiment / subject / study field / genre / main topic / spam / urgency / language.

Your output must always be a python list of strings only, with about 5 elements, and each element corresponds to a distinct text classification task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_brainstorm_s2s_tasks_prompt_with_topic(topic) -> List[Dict]:
    prompt = f"""Brainstorm a list of text matching tasks for the topic: {topic}, where both the queries and the groundtruth documents are very short (one or two sentences, even a short phrase).  

Here are a few examples:
- Given a scientific paper title, retrieve the title of papers that cite the given paper.
- Match a word with its definition.
- Provided a notable person's name, identify their occupation or achievement.

Your output must always be a python list of strings only, with about 5 elements, and each element corresponds to a distinct task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages

def get_brainstorm_p2p_tasks_prompt_with_topic(topic) -> List[Dict]:
    prompt = f"""Brainstorm a list of text matching tasks for the topic: {topic}, where the queries are long documents.  
  
Here are a few examples:  
- Given a document that supports a debatable argument, find another document that contains opposite arguments.  
- Provided a lengthy business proposal, retrieve competitive business strategies in the same industry.

Your output must always be a python list of strings only, with about 5 elements, and each element corresponds to a distinct task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    messages: List[Dict] = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages
