# import typer
from typing import Optional
import os
# import logging
import time
from datetime import datetime
from pytz import timezone
import logging


# import torch
# # import numpy as np
# from torch import nn
# from torch.optim import Adam, AdamW
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
# from datasets import load_dataset, load_metric
# import wandb
# from tqdm import tqdm
import json
import copy

# !pip install openai
import openai
print('after openai')
import re
# import wandb
# print('past imports')

data_path = '/data/shankar/ping_knowledge_injection/data/ecbd/ecbd_witheld.json'
def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]
    
def generate_augmentations(path):
    perplexities = []
    dataset = load_json(path)
    entities = {}
    for elem in dataset:
        if elem['ent_str'] not in entities.keys():
            entities[elem['ent_str']]=[elem]
        else:
            entities[elem['ent_str']].append(elem)
    # print(entities)
    ex={}
    augmented=[]
    with open('/data/shankar/ecbd_witheld_augmentations.json', 'w') as f:
        for key in entities.keys():
            elem = entities[key][0]
            gpt_prompt = 'Create a sentence extending the following prompt, and make sure that '+elem['ent_str']+' is somewhere in each of the generated sentences: '+elem['context']
            print('PROMPT')
            print(gpt_prompt)
            # print(blob)
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=gpt_prompt,
                temperature=0.9,
                n=15,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
            sentences = [choice.text.strip() for choice in response.choices]
            print(sentences)
            for elem in entities[key]:
                elem['augmented_probes'] = sentences
                # print('ELEM')
                print(elem)
                f.write(json.dumps(elem)+'\n')

    #     augmented.append(response['choices'][0][text])
    # elem['augmented_probes'] = augmented

        
		# print(generation)
    # print(sum(perplexities)/len(perplexities))

generate_augmentations(data_path)

