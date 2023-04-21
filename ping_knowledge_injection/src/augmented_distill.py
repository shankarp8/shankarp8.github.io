import typer
from typing import Optional
import os
import logging
import time
from datetime import datetime
from pytz import timezone
import logging

import torch
import numpy as np
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import load_dataset, load_metric
import wandb
from tqdm import tqdm
import json
import copy

from ping_data import LanguageModelingDataset, StandardDataset, QuestionGenerationDataset, RandomMatrixDataset, RandomQuestionDataset
from ping_utils import Method, Dataset, Optimizer
from ping_utils import validate_arguments, parse_valid_file
from ping_utils import DataCollatorForSeq2Seq
from ping_utils import converter


def curricula_distill(
    model, 
    teacher_type, 
    teacher, 
    tokenizer, 
    example, 
    device, 
    dataset_name, 
    initial_noise=0.15,
    final_noise=0.7, 
    top_p=1.0,
    repetition_penalty=1.0, 
    top_k=None,
    max_length=None, 
    length_penalty=1.0, 
    # teacher_path=teacher_path,
    # input_generator_path=input_generator_path,
    sample_temperature=1.0,
    num_steps = 5,
    num_samples = 0,
    gradient_accumulation_steps = 1,
    lr = 1e-4,
    optimizer_name = Optimizer.adamw,
    seed = 2022,
    log_every_steps = 100,
    valid_every_steps = 100,
    batch_size = 8, 
):
    # print('RANK!')
    # print(rank)
    # setup(rank, world_size, port)
    set_seed(seed)
    # set_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    # teacher = AutoModelForSeq2SeqLM.from_pretrained('t5-large')
    # teacher.to(device)

    # prompt = context
    # print('PROMPT')
    # print(prompt)
    # print()

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "t5-large",  # Temporary fix  # model_path
    #     use_fast=False,  # To prevent multiprocessing warning. cf) https://stackoverflow.com/a/67254879
    # )
    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))

    results = {}
    prompt = example['context']
    augmented_probes = example['augmented_probes']


    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

    num_generations = batch_size * gradient_accumulation_steps * num_steps
    for elem in augmented_probes:
        train_dataset = LanguageModelingDataset(prompt, tokenizer, num_generations, initial_noise=0.0, final_noise=0.0)
        print('PROMPT')
        print(prompt)
        print()
        train_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        train_dataset2 = LanguageModelingDataset(elem, tokenizer, num_generations, initial_noise, final_noise)
        train_dataloader2 = DataLoader(
            train_dataset2,
            batch_size=batch_size,
            collate_fn=train_data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        print('PROBE')
        print(elem)
        print()
        train_iterator = iter(train_dataloader)
        train_iterator2 = iter(train_dataloader2)

        kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
        
        iteration_step = 0
        optimization_step = 0

        while True:


            model.train()

            batch = next(train_iterator)
            batch2 = next(train_iterator2)
            batch = {k: v.to(device) for k, v in batch.items()}
            batch2 = {k: v.to(device) for k, v in batch2.items()}
            
            with torch.no_grad():

                teacher_input_ids = torch.cat([batch["input_ids"], batch2["input_ids"]],  dim=1)
                print('INPUTS')
                print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
                print()
                print(tokenizer.decode(batch2['input_ids'][0], skip_special_tokens=True))
                teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)

                teacher_outputs = teacher.generate(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_mask,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    do_sample=True,
                    temperature=sample_temperature, 
                    top_p=top_p,
                    max_length=max_length, 
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                )
                print('teacher output!')
                print(tokenizer.decode(teacher_outputs[0][0], skip_special_tokens=True))


                teacher_scores = []
                for position in teacher_outputs["scores"]:
                    teacher_scores.append(position)
                teacher_scores = torch.stack(teacher_scores, dim=1)



            student_input_ids = torch.cat([batch2['input_ids']], dim=1)

            student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

            input_ids_copy=copy.deepcopy(student_input_ids)
            decoder_input_ids_copy=copy.deepcopy(teacher_outputs["sequences"][:, :-1])
            attention_mask_copy=copy.deepcopy(student_mask)

            student_outputs = model(
                input_ids=input_ids_copy,
                decoder_input_ids=decoder_input_ids_copy,
                attention_mask=attention_mask_copy,
                output_hidden_states=True)
            labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
            labels[labels == tokenizer.pad_token_id] = -100
            logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)

            student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
            teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask

            temperature = 2
            loss_ce = (
                kl_criterion(
                    nn.functional.log_softmax(student_logits_selected / temperature, dim=-1),
                    nn.functional.softmax(teacher_logits_selected / temperature, dim=-1),
                )
                * (temperature) ** 2
            )
            loss = loss_ce
                
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            iteration_step += 1

            optimizer.step()
            optimizer.zero_grad()
            optimization_step += 1


            if optimization_step == num_steps:
                break

    curricula = True
    return model

def masked_distill(
    model, 
    teacher_type, 
    teacher, 
    tokenizer, 
    example, 
    device, 
    dataset_name, 
    initial_noise=0.15,
    final_noise=0.7, 
    top_p=1.0,
    repetition_penalty=1.0, 
    top_k=None,
    max_length=None, 
    length_penalty=1.0, 
    # teacher_path=teacher_path,
    # input_generator_path=input_generator_path,
    sample_temperature=1.0,
    num_steps = 5,
    num_samples = 0,
    gradient_accumulation_steps = 1,
    lr = 1e-4,
    optimizer_name = Optimizer.adamw,
    seed = 2022,
    log_every_steps = 100,
    valid_every_steps = 100,
    batch_size = 8, 
):
    # print('RANK!')
    # print(rank)
    # setup(rank, world_size, port)
    set_seed(seed)
    # set_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    # teacher = AutoModelForSeq2SeqLM.from_pretrained('t5-large')
    # teacher.to(device)

    # prompt = context
    # print('PROMPT')
    # print(prompt)
    # print()

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "t5-large",  # Temporary fix  # model_path
    #     use_fast=False,  # To prevent multiprocessing warning. cf) https://stackoverflow.com/a/67254879
    # )
    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))

    results = {}
    prompt = example['context']
    augmented_probes = example['augmented_probes']


    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

    num_generations = batch_size * gradient_accumulation_steps * num_steps
    for elem in augmented_probes:
        train_dataset = LanguageModelingDataset(prompt, tokenizer, num_generations, initial_noise=0.0, final_noise=0.0)
        print('PROMPT')
        print(prompt)
        print()
        train_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        train_dataset2 = LanguageModelingDataset(elem, tokenizer, num_generations, initial_noise, final_noise)
        train_dataloader2 = DataLoader(
            train_dataset2,
            batch_size=batch_size,
            collate_fn=train_data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        print('PROBE')
        print(elem)
        print()
        train_iterator = iter(train_dataloader)
        train_iterator2 = iter(train_dataloader2)

        kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
        
        iteration_step = 0
        optimization_step = 0

        while True:


            model.train()

            batch = next(train_iterator)
            batch2 = next(train_iterator2)
            batch = {k: v.to(device) for k, v in batch.items()}
            batch2 = {k: v.to(device) for k, v in batch2.items()}
            
            with torch.no_grad():

                teacher_input_ids = torch.cat([batch["input_ids"], batch2["input_ids"]],  dim=1)
                print('INPUTS')
                print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
                print()
                print(tokenizer.decode(batch2['input_ids'][0], skip_special_tokens=True))
                teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)

                teacher_outputs = teacher.generate(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_mask,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    do_sample=True,
                    temperature=sample_temperature, 
                    top_p=top_p,
                    max_length=max_length, 
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                )
                print('teacher output!')
                print(tokenizer.decode(teacher_outputs[0][0], skip_special_tokens=True))


                teacher_scores = []
                for position in teacher_outputs["scores"]:
                    teacher_scores.append(position)
                teacher_scores = torch.stack(teacher_scores, dim=1)



            student_input_ids = torch.cat([batch2['input_ids']], dim=1)

            student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

            input_ids_copy=copy.deepcopy(student_input_ids)
            decoder_input_ids_copy=copy.deepcopy(teacher_outputs["sequences"][:, :-1])
            attention_mask_copy=copy.deepcopy(student_mask)

            student_outputs = model(
                input_ids=input_ids_copy,
                decoder_input_ids=decoder_input_ids_copy,
                attention_mask=attention_mask_copy,
                output_hidden_states=True)
            labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
            labels[labels == tokenizer.pad_token_id] = -100
            logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)

            student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
            teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask

            temperature = 1
            loss_ce = (
                kl_criterion(
                    nn.functional.log_softmax(student_logits_selected / temperature, dim=-1),
                    nn.functional.softmax(teacher_logits_selected / temperature, dim=-1),
                )
                * (temperature) ** 2
            )
            loss = loss_ce
                
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            iteration_step += 1

            optimizer.step()
            optimizer.zero_grad()
            optimization_step += 1


            if optimization_step == num_steps:
                break

    curricula = True
    return model
