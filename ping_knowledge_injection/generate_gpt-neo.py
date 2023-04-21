# import typer
from typing import Optional
import os
# import logging
import time
# import numpy as np
# import torch
from datetime import datetime
# from pytz import timezone
# import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

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
# import json
import copy

import re
# import wandb
# print('past imports')

data_path = '/data/shankar/ping_knowledge_injection/data/ecbd/ecbd_10_augmentations.json'
def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]
    

def generate_augmentations(path):
    perplexities = []
    dataset = load_json(path)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    entities = {}
    for elem in dataset:
        if elem['ent_str'] not in entities.keys():
            entities[elem['ent_str']]=[elem]
        else:
            entities[elem['ent_str']].append(elem)
    # tokenizer.pad_token_id=tokenizer.eos_token_id
    with open('/data/shankar/ecbd_gpt-neo-1.3b_augmentations_final_40_tok_2.json', 'w') as f:
        for key in entities.keys():
            elem = entities[key][0]
            # elem['augmented_probes']=[]
            # elem['masked_augmentations']=[]
            # elem['augment_labels']=[]
            prompt = elem['context']
            ent_str = elem['ent_str']
            print('PROMPT')
            print(prompt)
            augmented_probes = []
            for i in range(10):
                inputs = tokenizer(prompt, return_tensors='pt')           
                outputs = model.generate(**inputs, do_sample=True, temperature=1, max_length=len(inputs['input_ids'][0])+32, top_p=0.9)
                old_sentence = tokenizer.batch_decode(outputs)[0]
                old_sentence = old_sentence[len(prompt):]
                if ent_str not in old_sentence:
                    sentence = ent_str+' '+old_sentence
                else:
                    sentence = old_sentence
                print('SENTENCE')
                print(sentence)
                augmented_probes.append(sentence)

            for elem in entities[key]:
                elem['augmented_probes'] = augmented_probes
                # print('ELEM')
                print(elem)
                f.write(json.dumps(elem)+'\n')
                
                # elem['augmented_probes'].append(sentence)
            # f.write(json.dumps(elem)+'\n')
    

    #     augmented.append(response['choices'][0][text])
    # elem['augmented_probes'] = augmented

        
		# print(generation)
    # print(sum(perplexities)/len(perplexities))

generate_augmentations(data_path)
