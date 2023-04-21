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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z")
logging.Formatter.converter = converter
teacher_path='/data/shankar/ping_pd/output/chat_teacher_dir'
input_generator_path='/data/shankar/ping_pd/output/chat_input_generator'
def t5_distill(
    model, 
    teacher_type, 
    teacher, 
    tokenizer, 
    context, 
    probey,
    gold_label, 
    device, 
    dataset_name, 
    top_p=1.0,
    repetition_penalty=1.0, 
    top_k=None,
    max_length=None, 
    length_penalty=1.0, 
    teacher_path=teacher_path,
    input_generator_path=input_generator_path,
    softmax_temperature=2, 
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
    beam_search=False, 
):
    #print('in t5 distill')
    # print('RANK!')
    # print(rank)
    # setup(rank, world_size, port)
    set_seed(seed)
    if dataset_name == 'ecbd':
        teacher_path='/data/shankar/ping2/output/chat_teacher_dir'
        input_generator_path='/data/shankar/ping2/output/chat_input_generator'

    if top_p == None:
        top_p=1.0
    if repetition_penalty==None:
        repetition_penalty=1.0
    if top_k==None:
        top_k=50
    if max_length==None:
        max_length=20
    if length_penalty==None:
        length_penalty=1.0

    # set_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    prompt = context


    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))

    results = {}
    # for pid, prompt_valid_data in tqdm(valid_data.items()):

    # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation


    #print('PROMPT')
    #print(prompt)
    #print('PROBEY')
    #print(probey)
    #print('GOLD LABEL')
    #print(gold_label)
    full_prompt = prompt+" "+probey
    # print('TEACHER PROMPT')
    # print(full_prompt)
    prompt_inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    probey_inputs = tokenizer(probey, return_tensors="pt", return_attention_mask=True)
    full_inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True)
    gold_label_tok = tokenizer(gold_label, return_tensors="pt", return_attention_mask=True)

    kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
      
    iteration_step = 0
    optimization_step = 0

    while True:
        # print('hey')
        # print('PRE-EDIT LOGITS BEFORE TRAIN')
        # model_copy=copy.deepcopy(model)
        # print(model_copy(**ex).logits)
        # print('third checkpoint')
        # if model.training:
        #     print("Model is in training mode")
        # else:
        #     print("Model is not in training mode")


        model.train()

        prompt_inputs["input_ids"] = prompt_inputs["input_ids"].to(device)
        probey_inputs["input_ids"] = probey_inputs["input_ids"].to(device)
        full_inputs["input_ids"] = full_inputs["input_ids"].to(device)
        prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"].to(device)
        probey_inputs["attention_mask"] = probey_inputs["attention_mask"].to(device)
        full_inputs["attention_mask"] = full_inputs["attention_mask"].to(device)



        
        with torch.no_grad():

            # teacher_input_ids = torch.cat([generated_input, batch["input_ids"]], dim=1)
            teacher_input_ids = full_inputs["input_ids"].to(device)
            teacher_mask = full_inputs["attention_mask"].to(device)

            gold_label_inputs = gold_label_tok["input_ids"].to(device)
            teacher_outputs = teacher(input_ids=teacher_input_ids, 
                                      decoder_input_ids=None,
                                      labels=gold_label_inputs,
                                      attention_mask=teacher_mask, 
                                      output_hidden_states=True)
            teacher_logits = teacher_outputs.logits

            # print('decoded teacher output', tokenizer.batch_decode(teacher_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True))

            # student_decoder_ids = torch.tensor(teacher_logits.argmax(dim=-1))
            student_decoder_ids = teacher_logits.argmax(dim=-1).clone().detach()



        # student_input_ids = torch.cat([probey_inputs["input_ids"][:-1]], dim=1)
        student_input_ids = probey_inputs['input_ids']
        student_mask = probey_inputs["attention_mask"]
        # student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

        input_ids_copy=copy.deepcopy(student_input_ids)
        decoder_input_ids_copy=copy.deepcopy(student_decoder_ids)
        attention_mask_copy=copy.deepcopy(student_mask)

        student_outputs = model(
            input_ids=input_ids_copy,
            decoder_input_ids=None,
            labels=student_decoder_ids,
            attention_mask=attention_mask_copy,
            output_hidden_states=True)
        student_logits = student_outputs.logits
        # print('decoded student output', tokenizer.batch_decode(student_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True))

        labels = teacher_logits
        labels[labels == tokenizer.pad_token_id] = -100
        logits_mask = (labels>-1)

        # student_logits_selected = torch.masked_select(student_logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        # # student_logits_selected = student_logits_selected.view(-1, student_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        # teacher_logits_selected = torch.masked_select(teacher_logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        # teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask




        temperature = softmax_temperature
        loss_ce = (
            kl_criterion(
                nn.functional.log_softmax(student_logits / temperature, dim=-1),
                nn.functional.softmax(teacher_logits/ temperature, dim=-1),
            )
            * (temperature) ** 2
        )
        loss = loss_ce

        # print('LOSS')
        # print(loss)
        # print()
            
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        iteration_step += 1

        if iteration_step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimization_step += 1



        if optimization_step == num_steps:
            break

    curricula = True
    return model

# print('student_logit_shape', student_logits.shape)
# print('teacher_logit_shape', teacher_logits.shape)
# print(blob)
# labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
# labels[labels == tokenizer.pad_token_id] = -100
# logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)


# student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
# student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
# teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
# teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask


# Input augmented distillation with probe, definition

def distill(
    model, #student model
    teacher_type, 
    teacher, #teacher model
    tokenizer, #mutual tokenizer for student, teacher
    context, #definition sentence
    probey, #probe sentence with mask
    device, #CUDA device
    dataset_name, #not used in this method
    gold_label, #not used in this method
    top_p=1.0,
    repetition_penalty=1.0, 
    top_k=None,
    max_length=None, 
    length_penalty=1.0, 
    sample_temperature=1.0,
    softmax_temperature=2, 
    num_steps = 5,
    num_samples = 0,
    gradient_accumulation_steps = 1,
    lr = 1e-4,
    optimizer_name = Optimizer.adamw,
    seed = 2022,
    log_every_steps = 100,
    valid_every_steps = 100,
    batch_size = 8, 
    beam_search=False, 
):

    set_seed(seed)


    # set_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    prompt = context


    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))

    results = {}

    # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

    print('PROMPT')
    print(prompt)
    print('PROBE')
    print(probey)

    train_dataset = QuestionGenerationDataset(prompt, tokenizer, chat=False, teacher=True)
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
    train_dataset2 = QuestionGenerationDataset(probey, tokenizer, chat=False)
    train_dataloader2 = DataLoader(
        train_dataset2,
        batch_size=batch_size,
        collate_fn=train_data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
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

            teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)

            teacher_outputs = teacher.generate(
                input_ids=teacher_input_ids,
                attention_mask=teacher_mask,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=sample_temperature,
                do_sample=True, 
                early_stopping=True,
                top_p=top_p,
                max_length=max_length, 
                length_penalty=length_penalty, 
                repetition_penalty=repetition_penalty,
                top_k=top_k,
            )
            print('teacher output!')
            for i in range(len(teacher_outputs[0])):
                print(len(teacher_outputs[0][i]))
                print(tokenizer.decode(teacher_outputs[0][i], skip_special_tokens=True))


            teacher_scores = []
            for position in teacher_outputs["scores"]:
                teacher_scores.append(position)
            teacher_scores = torch.stack(teacher_scores, dim=1)


        student_input_ids = torch.cat([batch2['input_ids']], dim=1)
        student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

        input_ids_copy=copy.deepcopy(student_input_ids)
        decoder_input_ids_copy = copy.deepcopy(teacher_outputs['sequences'][:, 1:])
        attention_mask_copy=copy.deepcopy(student_mask)

        student_outputs = model(
            input_ids=input_ids_copy,
            # decoder_input_ids=decoder_input_ids_copy,
            labels=teacher_outputs['sequences'], 
            attention_mask=attention_mask_copy,
            output_hidden_states=True)
        labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
        labels[labels == tokenizer.pad_token_id] = -100
        logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)


        student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  
        student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask




        temperature = softmax_temperature
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

        # if iteration_step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        optimization_step += 1


        if optimization_step == num_steps:
            break

    curricula = True
    return model

#Input-augmented distillation with definition, probe, and trained teacher 
def trained_distill(
    model, 
    context, 
    probey,
    device, 
    dataset_name, 
    top_p=1.0,
    repetition_penalty=1.0, 
    top_k=None,
    max_length=None, 
    length_penalty=1.0, 
    teacher_path=teacher_path,
    input_generator_path=input_generator_path,
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
    if dataset_name == 'ecbd':
        teacher_path='/data/shankar/ping2/output/chat_teacher_dir'
    else:
        teacher_path='/data/shankar/ping_pd/output/chat_teacher_dir'


    if top_p == None:
        top_p=1.0
    if repetition_penalty==None:
        repetition_penalty=1.0
    if top_k==None:
        top_k=50
    if max_length==None:
        max_length=20
    if length_penalty==None:
        length_penalty=1.0

    # set_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    # if teacher_path is not None:
    #     teacher = AutoModelForSeq2SeqLM.from_pretrained(
    #         teacher_path,
    #     )
    #     teacher.to(device)
    # else:
    #     teacher = None

    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        teacher_path,
    )
    teacher.to(device)

    # student_model_test = AutoModelForSeq2SeqLM.from_pretrained('/data/shankar/ping/output/chat_student_dir')
    # student_model_test.to(device)
    # model = student_model_test
    # print('MODEL')
    # print(model)
    # print()
    # print('STUDENT MODEL')
    # print(student_model_test)
    # print(blob)
    prompt = context
    print('PROMPT')
    print(prompt)
    print()

    # if input_generator_path is not None:
    #     input_generator = AutoModelForSeq2SeqLM.from_pretrained(
    #         input_generator_path,
    #     )
    #     input_generator.to(device)
    # else:
    #     input_generator = None

    # Assume model, teacher, and input generator all use the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-large",  # Temporary fix  # model_path
        use_fast=False,  # To prevent multiprocessing warning. cf) https://stackoverflow.com/a/67254879
    )
    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))

    # effective_batch_size = batch_size*gradient_accumulation_steps
    # if num_steps is None:
    #     num_steps = int(num_samples / effective_batch_size)
    # if rank == 0:
    #     logging.info(f"Effective batch size (batch_size * num_devices * gradient_accumulation_steps): {effective_batch_size}")
    #     logging.info(f"Number of optimization steps: {num_steps}")

    # valid_data = parse_valid_file(valid_path)
    # metric = load_metric("squad")

    # run_dir = f"{output_dir}/{name}"
    # os.makedirs(run_dir, exist_ok=True)

    results = {}
    # for pid, prompt_valid_data in tqdm(valid_data.items()):

    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     model_path,
    # )
    # model.to(rank)
    # model = DDP(
    #     model, device_ids=[rank], find_unused_parameters=True if freeze_embeddings else False
    # )

    # if freeze_embeddings:
    #     model.module.encoder.embed_tokens.weight.requires_grad = False
    #     model.module.decoder.embed_tokens.weight.requires_grad = False
    #     model.module.lm_head.weight.requires_grad = False


    # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

    print('LEARNING RATE')
    print(lr)
    print()
    print('MAX_LENGTH')
    print(max_length)
    print()

    print('PROMPT')
    print(prompt)
    print('PROBEY')
    print(probey)

    train_dataset = QuestionGenerationDataset(prompt, tokenizer, chat=False, teacher=True)
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
    train_dataset2 = QuestionGenerationDataset(probey, tokenizer, chat=False)
    train_dataloader2 = DataLoader(
        train_dataset2,
        batch_size=batch_size,
        collate_fn=train_data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    train_iterator = iter(train_dataloader)
    train_iterator2 = iter(train_dataloader2)

    kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
      
    iteration_step = 0
    optimization_step = 0

    while True:
        # print('hey')
        # print('PRE-EDIT LOGITS BEFORE TRAIN')
        # model_copy=copy.deepcopy(model)
        # print(model_copy(**ex).logits)
        # print('third checkpoint')
        # if model.training:
        #     print("Model is in training mode")
        # else:
        #     print("Model is not in training mode")


        model.train()
        # print('fourth checkpoint')
        # if model.training:
        #     print("Model is in training mode")
        # else:
        #     print("Model is not in training mode")
        # print('PRE-EDIT LOGITS JUST AFTER TRAIN')
        # model_copy2=copy.deepcopy(model)
        # print(model_copy2(**ex).logits)

        batch = next(train_iterator)
        batch2 = next(train_iterator2)
        # print('ATTENTION MASK')
        # print(batch['attention_mask'])
        # print()
        batch = {k: v.to(device) for k, v in batch.items()}
        batch2 = {k: v.to(device) for k, v in batch2.items()}
        
        with torch.no_grad():
            # generation_outputs = input_generator.generate(
            #     input_ids=batch["input_ids"], 
            #     attention_mask=batch["attention_mask"],
            #     do_sample=True,  # This causes difference between sequences and argmax of outputs
            #     temperature=sample_temperature,
            #     return_dict_in_generate=True,
            #     max_length=input_max_length,
            # )
            # print('GENERATION_OUTPUTS')
            # print(generation_outputs)
            # print()
            # print(blob)  # Exclude SOS

            # teacher_input_ids = torch.cat([generated_input, batch["input_ids"]], dim=1)

            teacher_input_ids = torch.cat([batch["input_ids"], batch2["input_ids"]],  dim=1)
            # print('past choke point')
            # # print(blob)
            # print('prompt decode')
            # print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
            # print()
            # print('pseudo decode')
            # print(tokenizer.decode(batch2['input_ids'][0],skip_special_tokens=True))
            # print()
            # print(blob)
            # print('TEACHER INPUT SHAPE')
            # print(teacher_input_ids.shape)
            teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)
            # teacher_outputs = model(input_ids=teacher_input_ids, attention_mask=teacher_mask, decoder_input_ids=None, output_scores=True, return_dict=True,)
            print('PARAMETERS')
            print('TOP_P', top_p)
            print('MAX_LENGTH', max_length)
            print('REPETITION_PEN', repetition_penalty)
            print('TOP_K', top_k)
            print('LENGTH_PEN', length_penalty)
            print('SAMPLE_TEMPERATURE',sample_temperature)
            

            teacher_outputs = teacher.generate(
                input_ids=teacher_input_ids,
                attention_mask=teacher_mask,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=sample_temperature, 
                do_sample=True,
                early_stopping=True,
                top_p=top_p,
                max_length=max_length, 
                repetition_penalty=repetition_penalty,
                top_k=top_k,
            )
            print('teacher output!')
            print(tokenizer.decode(teacher_outputs[0][0], skip_special_tokens=True))
            # # print(blob)

            teacher_scores = []
            for position in teacher_outputs["scores"]:
                teacher_scores.append(position)
            teacher_scores = torch.stack(teacher_scores, dim=1)
            # print('TEACHER SCORES SHAPE')
            # print(teacher_scores.shape)


        student_input_ids = torch.cat([batch2['input_ids']], dim=1)
        # print('STUDENT INPUT SHAPE')
        # print(student_input_ids.shape)
        student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

        # model_test=copy.deepcopy(model)
        input_ids_copy=copy.deepcopy(student_input_ids)
        decoder_input_ids_copy=copy.deepcopy(teacher_outputs["sequences"][:, :-1])
        attention_mask_copy=copy.deepcopy(student_mask)

        # student_outputs = model(
        #     input_ids=input_ids_copy,
        #     decoder_input_ids=decoder_input_ids_copy,
        #     attention_mask=attention_mask_copy,
        #     output_hidden_states=True)
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
        # print('LOSS')
        # print(loss)
        # print()
            
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        # print('PRE-EDIT LOGITS')
        # model_copy3=copy.deepcopy(model)
        # print(model_copy3(**ex).logits)
        # print()
        # original_params=copy.deepcopy(model.state_dict())
        # def compare_models(model1, model2):
        #     for param1, param2 in zip(model1.parameters(), model2.parameters()):
        #         if not torch.equal(param1, param2):
        #             return False
        #     return True
        loss.backward()
        iteration_step += 1

        # if iteration_step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        optimization_step += 1
        # print('final checkpoint')
        # if model.training:
        #     print("Model is in training mode")
        # else:
        #     print("Model is not in training mode")


        if optimization_step == num_steps:
            break

    curricula = True
    return model

#Vanilla distillation with larger teacher model
def vanilla_distill(
    model, 
    teacher,
    student_tokenizer,
    teacher_tokenizer, 
    probey,
    dataset_name,
    device, 
    top_p=1.0,
    repetition_penalty=1.0, 
    top_k=None,
    max_length=None, 
    length_penalty=1.0, 
    teacher_path=teacher_path,
    input_generator_path=input_generator_path,
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
    set_seed(seed)
    if dataset_name == 'ecbd':
        teacher_path='/data/shankar/ping2/output/chat_teacher_dir'
        input_generator_path='/data/shankar/ping2/output/chat_input_generator'

    if top_p == None:
        top_p=1.0
    if repetition_penalty==None:
        repetition_penalty=1.0
    if top_k==None:
        top_k=50
    if max_length==None:
        max_length=20
    if length_penalty==None:
        length_penalty=1.0

    torch.autograd.set_detect_anomaly(True)

    # teacher = AutoModelForSeq2SeqLM.from_pretrained('t5-3b')
    # teacher.to(device)
    prompt = probey
    print('PROMPT')
    print(prompt)
    print()

    # student_tokenizer = AutoTokenizer.from_pretrained(
    #     "t5-large",  # Temporary fix  # model_path
    #     use_fast=False,  # To prevent multiprocessing warning. cf) https://stackoverflow.com/a/67254879
    # )
    # teacher_tokenizer = AutoTokenizer.from_pretrained(
    #     "t5-3b",  # Temporary fix  # model_path
    #     use_fast=False,  # To prevent multiprocessing warning. cf) https://stackoverflow.com/a/67254879
    # )
    model.resize_token_embeddings(len(student_tokenizer))
    teacher.resize_token_embeddings(len(teacher_tokenizer))
    teacher.eval()


    results = {}


    # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

    print('LEARNING RATE')
    print(lr)
    print()
    print('MAX_LENGTH')
    print(max_length)
    print()

    print('PROMPT')
    print(probey)

    train_dataset = QuestionGenerationDataset(prompt, teacher_tokenizer, chat=False, teacher=True)
    train_data_collator = DataCollatorForSeq2Seq(
        teacher_tokenizer,
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
    train_iterator = iter(train_dataloader)
    # train_iterator2 = iter(train_dataloader2)

    kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
      
    iteration_step = 0
    optimization_step = 0

    while True:
        model.train()
        batch = next(train_iterator)
        # batch2 = next(train_iterator2)
        # print('ATTENTION MASK')
        # print(batch['attention_mask'])
        # print()
        batch = {k: v.to(device) for k, v in batch.items()}
        # batch2 = {k: v.to(device) for k, v in batch2.items()}
        
        with torch.no_grad():

            teacher_input_ids = torch.cat([batch["input_ids"]],  dim=1)

            print('prompt decode')
            print(teacher_tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
            print()
            teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)
            # teacher_outputs = model(input_ids=teacher_input_ids, attention_mask=teacher_mask, decoder_input_ids=None, output_scores=True, return_dict=True,)
            print('PARAMETERS')
            print('TOP_P', top_p)
            print('MAX_LENGTH', max_length)
            print('REPETITION_PEN', repetition_penalty)
            print('TOP_K', top_k)
            print('LENGTH_PEN', length_penalty)
            

            teacher_outputs = teacher.generate(
                input_ids=teacher_input_ids,
                attention_mask=teacher_mask,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                top_p=top_p,
                max_length=max_length, 
                repetition_penalty=repetition_penalty,
                top_k=top_k,
            )
            print('teacher output!')
            print(teacher_tokenizer.decode(teacher_outputs[0][0], skip_special_tokens=True))

            teacher_scores = []
            for position in teacher_outputs["scores"]:
                teacher_scores.append(position)
            teacher_scores = torch.stack(teacher_scores, dim=1)

        student_input_ids = torch.cat([batch['input_ids']], dim=1)
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
        labels[labels == student_tokenizer.pad_token_id] = -100
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
        # print('LOSS')
        # print(loss)
        # print()
            
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        iteration_step += 1

        # if iteration_step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        optimization_step += 1

        if optimization_step == num_steps:
            break

    curricula = True
    return model
# def compute_loss_beam_search(teacher, teacher_input_ids, teacher_mask, temperature, top_p, max_length, repetition_penalty, top_k, num_beams, model, batch_size, tokenizer, batch2):
#     teacher_output_sequences = teacher.generate(
#         input_ids=teacher_input_ids,
#         attention_mask=teacher_mask,
#         output_scores=True,
#         return_dict_in_generate=True,
#         temperature=temperature,
#         early_stopping=True,
#         top_p=top_p,
#         max_length=max_length, 
#         repetition_penalty=repetition_penalty,
#         top_k=top_k,
#         num_beams=num_beams
#     )

#     # Extract the top k sequences for each beam
#     top_k_sequences = teacher_output_sequences["sequences"]

#     # Print each output of the beam search by the teacher
#     for i in range(batch_size):
#         print(f"Example {i}:")
#         for j in range(num_beams):
#             sequence = top_k_sequences[i][j]
#             decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
#             print(f"Beam {j}: {decoded_sequence}")
#         print()
        
#     # Extract the teacher scores for each position in the sequence
#     teacher_scores = []
#     for position in teacher_output_sequences["scores"]:
#         teacher_scores.append(position)
#     teacher_scores = torch.stack(teacher_scores, dim=1)

#     # Prepare inputs for the student model
#     student_input_ids = torch.cat([batch2['input_ids']], dim=1)
#     student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

#     # Generate outputs using the student model
#     input_ids_copy = copy.deepcopy(student_input_ids)
#     decoder_input_ids_copy = copy.deepcopy(top_k_sequences[:, :, :-1].reshape(batch_size * num_beams, -1))
#     attention_mask_copy = copy.deepcopy(student_mask.repeat_interleave(num_beams, dim=0))
#     student_output_sequences = model.generate(
#         input_ids=input_ids_copy.repeat_interleave(num_beams, dim=0),
#         attention_mask=attention_mask_copy,
#         decoder_input_ids=decoder_input_ids_copy,
#         max_length=max_length,
#     )

#     # Extract the student scores for each position in the sequence
#     student_scores = student_output_sequences["scores"]

#     # Compute the KL divergence loss
#     temperature = 1
#     kl_div_loss = kl_criterion(
#         nn.functional.log_softmax(student_scores / temperature, dim=-1),
#         nn.functional.softmax(teacher_scores / temperature, dim=-1),
#     )
#     kl_div_loss = kl_div_loss * (temperature ** 2)

#     # Compute the final loss
#     loss = kl_div_loss
#     return loss


# #without iterator
# def distill_new(
#     model, 
#     teacher_type, 
#     teacher, 
#     tokenizer, 
#     context, 
#     probey,
#     device, 
#     dataset_name, 
#     top_p=1.0,
#     repetition_penalty=1.0, 
#     top_k=None,
#     max_length=None, 
#     length_penalty=1.0, 
#     teacher_path=teacher_path,
#     input_generator_path=input_generator_path,
#     sample_temperature=1.0,
#     softmax_temperature=2.0,
#     num_steps = 5,
#     num_samples = 0,
#     gradient_accumulation_steps = 1,
#     lr = 1e-4,
#     optimizer_name = Optimizer.adamw,
#     seed = 2022,
#     log_every_steps = 100,
#     valid_every_steps = 100,
#     batch_size = 8, 
#     beam_search=False, 
# ):
#     print('in distill new')
#     # print('RANK!')
#     # print(rank)
#     # setup(rank, world_size, port)
#     set_seed(seed)
#     if dataset_name == 'ecbd':
#         teacher_path='/data/shankar/ping2/output/chat_teacher_dir'
#         input_generator_path='/data/shankar/ping2/output/chat_input_generator'

#     if top_p == None:
#         top_p=1.0
#     if repetition_penalty==None:
#         repetition_penalty=1.0
#     if top_k==None:
#         top_k=50
#     if max_length==None:
#         max_length=20
#     if length_penalty==None:
#         length_penalty=1.0

#     # set_seed(seed)
#     torch.autograd.set_detect_anomaly(True)

#     prompt = context


#     model.resize_token_embeddings(len(tokenizer))
#     teacher.resize_token_embeddings(len(tokenizer))

#     # effective_batch_size = batch_size*gradient_accumulation_steps
#     # if num_steps is None:
#     #     num_steps = int(num_samples / effective_batch_size)
#     # if rank == 0:
#     #     logging.info(f"Effective batch size (batch_size * num_devices * gradient_accumulation_steps): {effective_batch_size}")
#     #     logging.info(f"Number of optimization steps: {num_steps}")

#     # valid_data = parse_valid_file(valid_path)
#     # metric = load_metric("squad")

#     # run_dir = f"{output_dir}/{name}"
#     # os.makedirs(run_dir, exist_ok=True)

#     results = {}
#     # for pid, prompt_valid_data in tqdm(valid_data.items()):

#     # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
#     if optimizer_name == Optimizer.adam:
#         optimizer = Adam(model.parameters(), lr=lr)
#     elif optimizer_name == Optimizer.adamw:
#         optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

#     print('LEARNING RATE')
#     print(lr)
#     print()
#     print('MAX_LENGTH')
#     print(max_length)
#     print()

#     print('PROMPT')
#     print(prompt)
#     print('PROBEY')
#     print(probey)
#     prompt_inputs = tokenizer(prompt, return_tensors="pt")
#     probey_inputs = tokenizer(probey, return_tensors="pt")
#     full_inputs = tokenizer(prompt+" "+probey, return_tensors="pt")


#     # train_dataset = QuestionGenerationDataset(prompt, tokenizer, chat=False, teacher=True)
#     # train_data_collator = DataCollatorForSeq2Seq(
#     #     tokenizer,
#     #     model=model,
#     #     label_pad_token_id=-100,
#     #     pad_to_multiple_of=8,
#     # )
#     # train_dataloader = DataLoader(
#     #     train_dataset,
#     #     batch_size=batch_size,
#     #     collate_fn=train_data_collator,
#     #     drop_last=False,
#     #     num_workers=0,
#     #     pin_memory=True,
#     # )
#     # train_dataset2 = QuestionGenerationDataset(probey, tokenizer, chat=False)
#     # train_dataloader2 = DataLoader(
#     #     train_dataset2,
#     #     batch_size=batch_size,
#     #     collate_fn=train_data_collator,
#     #     drop_last=False,
#     #     num_workers=0,
#     #     pin_memory=True,
#     # )
#     # train_iterator = iter(train_dataloader)
#     # train_iterator2 = iter(train_dataloader2)

#     kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
      
#     iteration_step = 0
#     optimization_step = 0

#     while True:
#         # print('hey')
#         # print('PRE-EDIT LOGITS BEFORE TRAIN')
#         # model_copy=copy.deepcopy(model)
#         # print(model_copy(**ex).logits)
#         # print('third checkpoint')
#         # if model.training:
#         #     print("Model is in training mode")
#         # else:
#         #     print("Model is not in training mode")


#         model.train()
#         # print('fourth checkpoint')
#         # if model.training:
#         #     print("Model is in training mode")
#         # else:
#         #     print("Model is not in training mode")
#         # print('PRE-EDIT LOGITS JUST AFTER TRAIN')
#         # model_copy2=copy.deepcopy(model)
#         # print(model_copy2(**ex).logits)

#         # batch = next(train_iterator)
#         # batch2 = next(train_iterator2)
#         # # print('ATTENTION MASK')
#         # # print(batch['attention_mask'])
#         # # print()
#         # batch = {k: v.to(device) for k, v in batch.items()}
#         # batch2 = {k: v.to(device) for k, v in batch2.items()}
#         prompt_inputs["input_ids"] = prompt_inputs["input_ids"].to(device)
#         probey_inputs["input_ids"] = probey_inputs["input_ids"].to(device)
#         full_inputs["input_ids"] = full_inputs["input_ids"].to(device)
#         prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"].to(device)
#         probey_inputs["attention_mask"] = probey_inputs["attention_mask"].to(device)
#         full_inputs["attention_mask"] = full_inputs["attention_mask"].to(device)
#         # print('LENGTH OF TEACHER INPUTS')
#         # print(len(full_inputs['input_ids'][0]))
#         # print('LENGTH OF STUDENT INPUTS')
#         # print(len(probey_inputs['input_ids'][0]))


        
#         with torch.no_grad():

#             # teacher_input_ids = torch.cat([generated_input, batch["input_ids"]], dim=1)
#             teacher_input_ids = full_inputs["input_ids"].to(device)
#             teacher_mask = full_inputs["attention_mask"].to(device)

#             # teacher_input_ids = torch.cat([prompt_inputs["input_ids"][:-1], probey_inputs["input_ids"][:-1]], dim=1)
            
#             # teacher_mask = torch.cat([prompt_inputs["attention_mask"][:-1], probey_inputs["attention_mask"][:-1]], dim=1)

#             # teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)
#             # print('TEACHER_MASK')
#             # print(prompt, probey)
#             # print(teacher_mask)
#             # print(blob)
#             # teacher_outputs = model(input_ids=teacher_input_ids, attention_mask=teacher_mask, decoder_input_ids=None, output_scores=True, return_dict=True,)
#             # print('PARAMETERS')
#             # print('TOP_P', top_p)
#             # print('MAX_LENGTH', max_length)
#             # print('REPETITION_PEN', repetition_penalty)
#             # print('TOP_K', top_k)
#             # print('LENGTH_PEN', length_penalty)
#             # print('SAMPLE_TEMPERATURE', sample_temperature)

#             # if beam_search:
#             #     beam_loss = compute_loss_beam_search(teacher=teacher, teacher_input_ids=teacher_input_ids, teacher_mask=teacher_mask, 
#             #                                         temperature=sample_temperature, top_p=top_p, max_length=max_length, repetition_penalty=repetition_penalty, 
#             #                                         top_k=top_k, num_beams=5, model=model, batch_size=batch_size, tokenizer=tokenizer, batch2=batch2)
            
#             # for temp in [0.5, 1.0, 1.5, 2.0, 2.5, 10, 25]:
#             #     teacher_outputs = teacher.generate(
#             #         input_ids=teacher_input_ids,
#             #         attention_mask=teacher_mask,
#             #         output_scores=True,
#             #         return_dict_in_generate=True,
#             #         temperature=temp,
#             #         early_stopping=True,
#             #         top_p=top_p,
#             #         max_length=max_length, 
#             #         repetition_penalty=repetition_penalty,
#             #         top_k=top_k,
#             #     )
#             #     print(f"Generated sequence at temperature {temp}: {tokenizer.decode(teacher_outputs[0][0], skip_special_tokens=True)}")
#             # print(blob)
#             teacher_outputs = teacher.generate(
#                 input_ids=teacher_input_ids,
#                 attention_mask=teacher_mask,
#                 output_scores=True,
#                 return_dict_in_generate=True,
#                 temperature=sample_temperature,
#                 do_sample=True, 
#                 early_stopping=True,
#                 top_p=top_p,
#                 max_length=10, 
#                 length_penalty=length_penalty, 
#                 repetition_penalty=repetition_penalty,
#                 top_k=top_k,
#             )
#             print('teacher output!')
#             # print(len(teacher_outputs[0]))
#             for i in range(len(teacher_outputs[0])):
#                 print(len(teacher_outputs[0][i]))
#                 print(tokenizer.decode(teacher_outputs[0][i], skip_special_tokens=True))
#             # print(tokenizer.decode(teacher_outputs[0][1], skip_special_tokens=True))
#             # print(tokenizer.decode(teacher_outputs[0][1], skip_special_tokens=True))
#             # # print(blob)

#             teacher_scores = []
#             for position in teacher_outputs["scores"]:
#                 teacher_scores.append(position)
#                 # print('POSITION')
#                 # print(position)
#             teacher_scores = torch.stack(teacher_scores, dim=1)
#             # print('teacher scores')
#             # print(teacher_scores)
#             # print(blob)
#             # print(blob)
#             # print('TEACHER SCORES SHAPE')
#             # print(teacher_scores.shape)


#         # student_input_ids = torch.cat([probey_inputs["input_ids"][:-1]], dim=1)
#         student_input_ids = probey_inputs['input_ids']
#         student_mask = probey_inputs["attention_mask"]
#         # student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

#         input_ids_copy=copy.deepcopy(student_input_ids)
#         decoder_input_ids_copy=copy.deepcopy(teacher_outputs["sequences"][:, :-1])
#         attention_mask_copy=copy.deepcopy(student_mask)

#         student_outputs = model(
#             input_ids=input_ids_copy,
#             decoder_input_ids=decoder_input_ids_copy,
#             attention_mask=attention_mask_copy,
#             output_hidden_states=True)

#         labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
#         labels[labels == tokenizer.pad_token_id] = -100
#         logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)
#         # print('PRE-LENGTHS')
#         # print('student logit length', student_outputs.logits.shape)
#         # print('teacher logit length', teacher_scores.shape)

#         student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
#         student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
#         teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
#         teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
#         # print('POST-LENGTHS')
#         # print('student_logit_lengths', student_logits_selected.shape)
#         # print('teacher_logit_lengths', teacher_logits_selected.shape)
#         # print(blob)

#         temperature = softmax_temperature
#         loss_ce = (
#             kl_criterion(
#                 nn.functional.log_softmax(student_logits_selected / temperature, dim=-1),
#                 nn.functional.softmax(teacher_logits_selected / temperature, dim=-1),
#             )
#             * (temperature) ** 2
#         )
#         if not beam_search:
#             loss = loss_ce
#         else:
#             loss=beam_loss
#         # print('LOSS')
#         # print(loss)
#         # print()
            
#         if gradient_accumulation_steps > 1:
#             loss = loss / gradient_accumulation_steps

#         # print('PRE-EDIT LOGITS')
#         # model_copy3=copy.deepcopy(model)
#         # print(model_copy3(**ex).logits)
#         # print()
#         # original_params=copy.deepcopy(model.state_dict())
#         # def compare_models(model1, model2):
#         #     for param1, param2 in zip(model1.parameters(), model2.parameters()):
#         #         if not torch.equal(param1, param2):
#         #             return False
#         #     return True
#         loss.backward()
#         iteration_step += 1

#         # if iteration_step % gradient_accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()
#         optimization_step += 1
#         # print('final checkpoint')
#         # if model.training:
#         #     print("Model is in training mode")
#         # else:
#         #     print("Model is not in training mode")


#         if optimization_step == num_steps:
#             break

#     curricula = True
#     return model
