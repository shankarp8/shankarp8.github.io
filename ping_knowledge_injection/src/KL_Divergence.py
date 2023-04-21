import typer
from typing import Optional
import os
import logging
import time
from datetime import datetime
from pytz import timezone
import logging
import sys
# sys.path.append('/data/shankar/ping_knowledge_injection')

import torch
import numpy as np
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import load_dataset, load_metric
import wandb
from tqdm import tqdm
import json
import copy

# from ping_data import LanguageModelingDataset, StandardDataset, QuestionGenerationDataset, RandomMatrixDataset, RandomQuestionDataset
from ping_utils import Method, Dataset, Optimizer
# from ping_utils import validate_arguments, parse_valid_file
# from ping_utils import DataCollatorForSeq2Seq
# from ping_utils import converter


# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z")
# logging.Formatter.converter = converter
# teacher_path='/data/shankar/ping_pd/output/chat_teacher_dir'
# input_generator_path='/data/shankar/ping_pd/output/chat_input_generator'


import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
path = '/data/shankar/ping_knowledge_injection/data/ecbd/ecbd_random_augmentations.json'


def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]

def compute_all(path):
    device = torch.device('cuda:1')
    model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    teacher = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = model.to(device)
    teacher = teacher.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_json(path)
    with open('/data/shankar/ping_knowledge_injection/divergences_random.txt', 'w') as f:
        entities = {}
        for elem in dataset:
            if elem['ent_str'] not in entities.keys():
                entities[elem['ent_str']]=[elem]
            else:
                entities[elem['ent_str']].append(elem)
        for key in entities.keys():
            elem = entities[key][0]
            for probe in elem['augmented_probes']:
                divergence = compute_divergence(model=model, teacher=teacher, tokenizer=tokenizer, context=elem['context'], 
                                                probey=probe,device=device)
                f.write(elem['context'])
                f.write('\n')
                # f.write(probe)
                f.write('\n')
                f.write('Per-token KL Loss : {}\n\n'.format('  '.join(
                        ['{} ({:.2f})'.format(token, loss) for token, loss in
                        divergence[0]])))
                f.write('\n')
                f.write('\n')
        
def compute_divergence(
    model, 
    # teacher_type, 
    teacher, 
    tokenizer, 
    context, 
    probey,
    device, 
    # dataset_name, 
    top_p=1.0,
    repetition_penalty=1.0, 
    top_k=None,
    max_length=None, 
    length_penalty=1.0, 
    # teacher_path=teacher_path,
    # input_generator_path=input_generator_path,
    sample_temperature=1.0,
    softmax_temperature=1.0, 
    num_steps=5,
    num_samples=0,
    gradient_accumulation_steps=2,
    lr=1e-4,
    optimizer_name=Optimizer.adamw,
    seed=2022,
    log_every_steps=100,
    valid_every_steps=100,
    batch_size=8, 
    beam_search=False, 
):
    set_seed(seed)


    if top_p is None:
        top_p = 1.0
    if repetition_penalty is None:
        repetition_penalty = 1.0
    if top_k is None:
        top_k = 50
    if max_length is None:
        max_length = 16
    if length_penalty is None:
        length_penalty = 1.0

    torch.autograd.set_detect_anomaly(True)



    prompt = context
    teacher_context = prompt+' '+probey
    prompt_inputs = tokenizer(prompt, return_tensors='pt')
    probey_inputs = tokenizer(probey, return_tensors='pt')
    teacher_inputs = tokenizer(teacher_context, return_tensors='pt')
    prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
    probey_inputs['input_ids'] = probey_inputs['input_ids'].to(device)

    prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
    probey_inputs['attention_mask'] = probey_inputs['attention_mask'].to(device)

    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))
    # tokenizer.pad_token = tokenizer.eos_token

    optimizer = None
    # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)



    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    iteration_step = 0
    optimization_step = 0

    while True:

        model.train()


        with torch.no_grad():
            teacher_input_ids = torch.cat([prompt_inputs['input_ids'], probey_inputs['input_ids']], dim=1)
            teacher_mask = torch.ones_like(teacher_input_ids)
            teacher_outputs = teacher(teacher_input_ids, attention_mask=teacher_mask)

        # Train the student model to match the teacher logits
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        optimizer.zero_grad()

        # Generate logits for the student model
        student_input_ids = probey_inputs['input_ids']
        student_mask = probey_inputs['attention_mask']
        student_logits = model(student_input_ids, attention_mask=student_mask).logits

        teacher_logits_selected = teacher_outputs.logits[:, -student_logits.shape[1]:-1, :]

        student_logits_selected = student_logits[:, :-1, :]


        # Calculate the distillation loss
        temperature = softmax_temperature
        loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student_logits_selected/temperature, dim=-1), 
                                                    torch.nn.functional.softmax(teacher_logits_selected/temperature, dim=-1), 
                                                    reduction='none').sum(dim=-1) * (temperature ** 2)
        token_ids = student_input_ids.detach().cpu().numpy()
        token_kl_loss_array = loss.detach().cpu().numpy()

        token_kl_loss_pairs = []
        for i in range(token_ids.shape[0]):  
            token_loss_pairs = []
            for j in range(token_kl_loss_array.shape[1]):  
                token = tokenizer.decode([token_ids[i, j]], clean_up_tokenization_spaces=True)
                kl_loss = token_kl_loss_array[i, j]
                token_loss_pairs.append((token, kl_loss))
            token_kl_loss_pairs.append(token_loss_pairs)
            
        return token_kl_loss_pairs
    
compute_all(path)