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

from ping_data import LanguageModelingDataset, StandardDataset, QuestionGenerationDataset, RandomMatrixDataset, RandomQuestionDataset
from ping_utils import Method, Dataset, Optimizer
from ping_utils import validate_arguments, parse_valid_file
from ping_utils import DataCollatorForSeq2Seq
from ping_utils import converter


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z")
logging.Formatter.converter = converter



def curricula(
    model, 
    context, 
    device, 
    freeze_embeddings = False,
    initial_noise = 0.15,
    final_noise = 0.7,
    input_max_length = 30,
    num_steps = 5,
    num_samples = 0,
    gradient_accumulation_steps = 1,
    lr = 1e-4,
    optimizer_name = Optimizer.adamw,
    seed = 42,
    log_every_steps = 100,
    valid_every_steps = 100,
    batch_size = 8, 
):

    set_seed(seed)

    # Assume model, teacher, and input generator all use the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-large",  # Temporary fix  # model_path
        use_fast=False,  # To prevent multiprocessing warning. cf) https://stackoverflow.com/a/67254879
    )

    # effective_batch_size = batch_size * torch.cuda.device_count() * gradient_accumulation_steps
    if num_steps is None:
        num_steps = int(num_samples / effective_batch_size)
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

    if freeze_embeddings:
        model.module.encoder.embed_tokens.weight.requires_grad = False
        model.module.decoder.embed_tokens.weight.requires_grad = False
        model.module.lm_head.weight.requires_grad = False

    prompt = context
    # print('EARLIER PROMPT')
    # print(prompt)
    # valid_dataset = StandardDataset(prompt_valid_data, tokenizer, chat, with_prompt)
    # DataCollatorForSeq2Seq produces decoder_input_ids from labels
    # valid_data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=-100,
    #     pad_to_multiple_of=8,
    # )
    # valid_dataloader = DataLoader(
    #     valid_dataset,
    #     batch_size=batch_size,
    #     collate_fn=valid_data_collator,
    #     drop_last=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     shuffle=False,
    #     sampler=None,
    # )

    # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
    if optimizer_name == Optimizer.adam:
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == Optimizer.adamw:
        optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

    # if rank == 0:
    #     wandb.init(
    #         project="persona",
    #         name=f"{name}/{pid}",
    #         dir=run_dir,
    #         config={
    #             "method": method.value,
    #             "dataset": dataset.value,
    #             "chat": chat,
    #             "with_prompt": with_prompt,
    #             "valid-path": valid_path,
    #             "model-path": model_path,
    #             "teacher-path": teacher_path,
    #             "input-generator-path": input_generator_path,
    #             "freeze-embeddings": freeze_embeddings,
    #             "sample_temperature": sample_temperature,
    #             "input-max-length": input_max_length,
    #             "num_steps": num_steps,
    #             "num_samples": num_samples,
    #             "batch_size": batch_size,
    #             "gradient-accumulation-steps": gradient_accumulation_steps,
    #             "lr": lr,
    #             "optimizer_name": optimizer_name.value,
    #             "seed": seed,
    #         },
    #         reinit=True,
    #     )
    # device = torch.device('cuda:0')


    num_generations = batch_size * gradient_accumulation_steps * num_steps
    train_dataset = LanguageModelingDataset(prompt, tokenizer, num_generations, initial_noise, final_noise)
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
    train_iterator = iter(train_dataloader)
    iteration_step = 0
    optimization_step = 0

    while True:
        model.train()

        batch = next(train_iterator)
        batch = {k: v.to(device) for k, v in batch.items()}
        print('CURRENT INPUT')
        print(tokenizer.decode(batch["input_ids"][0]))
        print()
        # print('BATCH DECODE')
        # print(tokenizer.batch_decode(batch["input_ids"]))

        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        curricula = True

            

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        iteration_step += 1

        if iteration_step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimization_step += 1

            # if rank == 0:
            #     wandb.log(
            #         {
            #             "train/loss": loss.item() * gradient_accumulation_steps,
            #         },
            #         step=optimization_step,
            #     )

        if optimization_step == num_steps:
            break

    #     if iteration_step % (gradient_accumulation_steps * valid_every_steps) == 0:
    #         dist.barrier()
    

    # dist.barrier()
    
    # dist.barrier()
    return model
    # cleanup()


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


