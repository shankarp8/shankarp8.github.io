from typing import Dict, List, Any
import numpy as np

import torch
import torch.nn.functional as F
# from .rome.util import nethook
# from .rome.util.generate import generate_interactive, generate_fast
import sys
sys.path.insert(1, '/data/shankar/ping_knowledge_injection/src')
# from .rome.experiments.py.demo import demo_model_editing
import copy
import random
from curricula import curricula
from input_generation import ping
from distill import distill, trained_distill, vanilla_distill, t5_distill #,distill_new
from augmented_distill import curricula_distill, masked_distill
from distill_gpt import gpt_distill, gpt_distill_generate, gpt_distill_next_token, gpt_distill_after_entity_span
'''
https://github.com/eric-mitchell/mend
'''

def get_log_probs(pred, targ, shift=False):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(-1)  # We want to get the whole sequence right
    acc = correct.float().mean()

    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    log_prob_all = unmasked_log_probs * mask.float()
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": -log_prob,
        "log_prob_all": log_prob_all
    }


def compute_specificity_entity_inferences(model_raw, model_ft,
                                          specificity_batches, shift=False):
    pre_loc_dicts = []
    post_loc_dicts = []
    is_mend = hasattr(model_raw, 'model')
    name_or_path = model_raw.model.name_or_path if is_mend else \
        model_raw.name_or_path
    for s_batch in specificity_batches:
        if 't5' in name_or_path:
            ex = s_batch["edit_inner"][0]['probe_sentence']
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['decoder_attention_mask'] = s_batch["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
        else:
            ex = {}
            ex['input_ids'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = s_batch["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
        pre_loc_logits = model_raw(**ex) if is_mend else model_raw(**ex).logits
        post_loc_logits = model_ft(**ex) if is_mend else model_ft(**ex).logits
        # print('HEREEEEEE')
        # print(torch.eq(pre_loc_logits, post_loc_logits))
        pre_loc_dict = []
        post_loc_dict = []
        n_probe_labels = s_batch['edit_inner'][0]['labels'][
            'input_ids'].size(0)
        for i in range(n_probe_labels):
            label = s_batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_loc_dict.append(get_log_probs(
                pre_loc_logits,label, shift=shift))
            post_loc_dict.append(get_log_probs(
                post_loc_logits, label, shift=shift))
           # print('ACCURACY')
           # print(pre_loc_dict[i]['acc'])
         #   print(post_loc_dict[i]['acc'])

        #    print(pre_loc_dict[i]['log_prob'])
       #     print(post_loc_dict[i]['log_prob'])
      #  print(kwa)
        pre_loc_dicts.append(pre_loc_dict)
        post_loc_dicts.append(post_loc_dict)
    # print('yo!')
    # print('\n')
    # #print(pre_loc_dicts)
    # print('\n')
    # print('yo!')
    #print(post_loc_dicts)
    # print(torch.eq(pre_loc_dicts, post_loc_dicts))
    #print(pre_loc_dicts==post_loc_dicts)
    return pre_loc_dicts, post_loc_dicts


def compute_specificity_ecbd(model_raw, model_ft, specificity_batches):
    pre_loc_logits = []
    post_loc_logits = []
    is_mend = hasattr(model_raw, 'model')
    name_or_path = model_raw.model.name_or_path if is_mend else \
        model_raw.name_or_path
    i = 0
    for s_batch in specificity_batches:
        if 't5' in name_or_path :
            ex = s_batch["edit_inner"][0]['probe_sentence']
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['decoder_attention_mask'] = s_batch["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
        else:
            ex = {}
            ex['input_ids'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = s_batch["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
        pre_loc_logits.append(
            model_raw(**ex) if is_mend else model_raw(**ex).logits)
        post_loc_logits.append(
            model_ft(**ex) if is_mend else model_ft(**ex).logits)
        i += 1
    return pre_loc_logits, post_loc_logits

def generate_sample(array, m, k):     
    random.seed(18)
    if not (0 < k <= m <= len(array)):
        raise ValueError("Invalid parameters. Ensure 0 < k <= m <= len(array)")

    unique_elements = random.sample(array, k)

    # Start with one instance of each unique element
    sample = unique_elements.copy()

    # Calculate remaining count
    remaining = m - k
    
    # Add remaining elements to the sample
    remaining_elements = [random.choice(unique_elements) for _ in range(remaining)]
    sample.extend(remaining_elements)
    
    # Shuffle the sample to randomize element order
    random.shuffle(sample)

    return sample


def null_t5(batch, model_ft, model_raw=None, specificity_batches=None,
          dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_raw, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_raw, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def curricula_distill_t5(batch, teacher_type, model_raw, teacher, tokenizer, example, device, initial_noise, final_noise, lr, num_steps, max_length, top_p, repetition_penalty, top_k, length_penalty, specificity_batches=None,
          dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    model_original = copy.deepcopy(model_raw)


    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits




    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    # print()
    # print('LEARNING_RATE!!!!', lr)
    # print('NUM_STEPS!!!', num_steps)

    model_ft = curricula_distill(model_raw, teacher_type, teacher=teacher, tokenizer=tokenizer, example=example, device=device, dataset_name=dataset_name, initial_noise=initial_noise, final_noise=final_noise,  lr=lr, num_steps=num_steps, max_length=max_length, top_p=top_p, 
                               repetition_penalty=repetition_penalty, top_k=top_k, length_penalty=length_penalty)
    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    # print('POST EDIT LOGITS')
    # print(post_edit_logits)
    # print(blob)
    # print()

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)



def vanilla_distill_t5(batch, model_raw, teacher, student_tokenizer, teacher_tokenizer, probey, device, lr, num_steps, max_length, top_p, repetition_penalty, top_k, length_penalty, specificity_batches=None,
          dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    model_original = copy.deepcopy(model_raw)
 

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits




    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    # print()
    # print('LEARNING_RATE!!!!', lr)
    # print('NUM_STEPS!!!', num_steps)

    model_ft = vanilla_distill(model_raw, teacher=teacher, student_tokenizer=student_tokenizer, teacher_tokenizer=teacher_tokenizer, 
                               probey=probey, device=device, dataset_name=dataset_name, lr=lr, num_steps=num_steps, max_length=max_length, top_p=top_p, 
                               repetition_penalty=repetition_penalty, top_k=top_k, length_penalty=length_penalty)
    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits



    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def ent_str_distill_gpt(batch, model_raw, teacher_type, teacher, tokenizer, context, probey, device, lr, num_steps, max_length, sample_temperature, softmax_temperature,  top_p, 
               repetition_penalty, top_k, length_penalty, beam_search, batch_size, gold_label, specificity_batches=None, dataset_name=None, ent_str_only='False'):
    
    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit

    model_original = copy.deepcopy(model_raw)


    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits



    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    model_ft = gpt_distill_generate(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, 
                        lr=lr, num_steps=num_steps, 
                        max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                        sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
    # else:
    #     model_ft = gpt_distill_next_token(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, 
    #                 lr=lr, num_steps=num_steps, 
    #                 max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
    #                 sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)

    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    # print('POST EDIT LOGITS')
    # print(post_edit_logits)
    # print(blob)
    # print()

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def distill_llama(batch, model_raw, teacher_type, teacher, tokenizer, context, probey, device, lr, num_steps, max_length, sample_temperature, softmax_temperature,  top_p, 
               repetition_penalty, top_k, length_penalty, beam_search, batch_size, gold_label, specificity_batches=None, dataset_name=None, ent_str_only='False'):
    
    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit

    model_original = copy.deepcopy(model_raw)


    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits



    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    model_ft = gpt_distill(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, 
                        lr=lr, num_steps=num_steps, 
                        max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                        sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
    # else:
    #     model_ft = gpt_distill_next_token(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, 
    #                 lr=lr, num_steps=num_steps, 
    #                 max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
    #                 sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)

    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    # print('POST EDIT LOGITS')
    # print(post_edit_logits)
    # print(blob)
    # print()

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def distill_gpt(batch, model_raw, teacher_type, teacher, tokenizer, context, probey, device, lr, num_steps, max_length, sample_temperature, softmax_temperature,  top_p, 
               repetition_penalty, top_k, length_penalty, beam_search, batch_size, gold_label, specificity_batches=None, dataset_name=None, ent_str_only='False'):
    
    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit

    model_original = copy.deepcopy(model_raw)


    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits



    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    model_ft = gpt_distill(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, 
                        lr=lr, num_steps=num_steps, 
                        max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                        sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
    # else:
    #     model_ft = gpt_distill_next_token(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, 
    #                 lr=lr, num_steps=num_steps, 
    #                 max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
    #                 sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)

    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits


    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def random_distill_gpt(batch, model_raw, teacher_type, teacher, unmasked_probes, tokenizer, context,  device, lr, num_steps, max_length, sample_temperature, softmax_temperature,  top_p, 
                   num_probes,  gold_labels, repetition_penalty, top_k, length_penalty, beam_search, num_updates, batch_size, ent_str, after_ent_span,specificity_batches=None, dataset_name=None):
    
    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit

    model_original = copy.deepcopy(model_raw)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    
    counter=0
    model_ft = copy.deepcopy(model_raw)
    for probey in unmasked_probes: #distill information from each probe to model
        model_raw = gpt_distill(model=model_ft, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer, context=context,ent_str=ent_str, 
                                                probey=probey, device=device, dataset_name=dataset_name, lr=lr, num_steps=num_steps, max_length=max_length, 
                                                top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                                                sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
        if model_raw!=False:
            model_ft = model_raw
            counter+=1
        if counter==num_updates:
            break

    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def ft_distill_llama(batch, model_raw, teacher_type, teacher, tokenizer, context, probes, unmasked_probes,  device, lr, num_steps, max_length, sample_temperature, softmax_temperature,  top_p, 
                   num_probes,  gold_labels, repetition_penalty, top_k, length_penalty, beam_search, num_updates, batch_size, ent_str, after_ent_span=None,specificity_batches=None, dataset_name=None):
    
    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit

    model_original = copy.deepcopy(model_raw)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits
    

    sample = generate_sample(unmasked_probes, num_updates, 5) #get random sample of num_probes unique UNMASKED probes
    # sample = unmasked_probes[:num_probes]
    # assert len(sample) == num_updates
    # uniques = []
    # for i in range(len(sample)):
    #     if sample[i] not in uniques:
    #         uniques.append(sample[i])
    # assert len(uniques) == num_probes
    # losses=[]
    # if not after_ent_span:
    #     for probey in sample: #distill information from each probe to model
    #         assert False
    #         model_ft = gpt_distill(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer, context=context,ent_str=ent_str, 
    #                                                 probey=probey, device=device, dataset_name=dataset_name, lr=lr, num_steps=num_steps, max_length=max_length, 
    #                                                 top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
    #                                                 sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
    #         model_raw = model_ft
    # else:
    counter=0
    model_ft = copy.deepcopy(model_raw)
    for probey in unmasked_probes: #distill information from each probe to model
        model_raw = gpt_distill(model=model_ft, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer, context=context,ent_str=ent_str, 
                                                probey=probey, device=device, dataset_name=dataset_name, lr=lr, num_steps=num_steps, max_length=max_length, 
                                                top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                                                sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
        if model_raw!=False:
            model_ft = model_raw
            counter+=1
        if counter==num_updates:
            break

    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def ft_distill_gpt(batch, model_raw, teacher_type, teacher, tokenizer, context, probes, unmasked_probes,  device, lr, num_steps, max_length, sample_temperature, softmax_temperature,  top_p, 
                   num_probes,  gold_labels, repetition_penalty, top_k, length_penalty, beam_search, num_updates, batch_size, ent_str, after_ent_span=None,specificity_batches=None, dataset_name=None):
    
    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit

    model_original = copy.deepcopy(model_raw)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits
    

    sample = generate_sample(unmasked_probes, num_updates, 5) #get random sample of num_probes unique UNMASKED probes
    # sample = unmasked_probes[:num_probes]
    # assert len(sample) == num_updates
    # uniques = []
    # for i in range(len(sample)):
    #     if sample[i] not in uniques:
    #         uniques.append(sample[i])
    # assert len(uniques) == num_probes
    # losses=[]
    # if not after_ent_span:
    #     for probey in sample: #distill information from each probe to model
    #         assert False
    #         model_ft = gpt_distill(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer, context=context,ent_str=ent_str, 
    #                                                 probey=probey, device=device, dataset_name=dataset_name, lr=lr, num_steps=num_steps, max_length=max_length, 
    #                                                 top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
    #                                                 sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
    #         model_raw = model_ft
    # else:
    counter=0
    model_ft = copy.deepcopy(model_raw)
    for probey in unmasked_probes: #distill information from each probe to model
        model_raw = gpt_distill(model=model_ft, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer, context=context,ent_str=ent_str, 
                                                probey=probey, device=device, dataset_name=dataset_name, lr=lr, num_steps=num_steps, max_length=max_length, 
                                                top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                                                sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
        if model_raw!=False:
            model_ft = model_raw
            counter+=1
        if counter==num_updates:
            break

    model_ft.eval()


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def ft_distill_t5(batch, model_raw, teacher_type, teacher, tokenizer, context, probes, unmasked_probes, gold_labels,  device, lr, num_steps, max_length, sample_temperature, top_p, softmax_temperature, num_probes,   
               repetition_penalty, top_k, length_penalty, beam_search, batch_size, num_updates, specificity_batches=None, dataset_name=None, ent_str=None, after_ent_span=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    model_original = copy.deepcopy(model_raw)


    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits



    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    # print()
    # print('LEARNING_RATE!!!!', lr)
    # print('NUM_STEPS!!!', num_steps)
    # print('in edit func')
    for i in range(len(probes)):
        model_ft = t5_distill(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probes[i], device=device, dataset_name=dataset_name, lr=lr, 
                       num_steps=num_steps, gold_label=gold_labels[i],
                       max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                       sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search, batch_size=batch_size)
        # print('ONE MINI ITERATION')
        model_raw = model_ft
    model_ft.eval()
    # print('ONE ITERATION IN EDIT FUNC!')


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    # print('POST EDIT LOGITS')
    # print(post_edit_logits)
    # print(blob)
    # print()

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def multiple_mask_distill_t5(batch, model_raw, teacher_type, teacher, tokenizer, context, probes, unmasked_probes, gold_labels,  device, lr, num_steps, max_length, 
                             sample_temperature, top_p, softmax_temperature, num_probes,repetition_penalty, top_k, length_penalty, beam_search, batch_size, 
                             num_updates, specificity_batches=None, dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    model_original = copy.deepcopy(model_raw)


    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits



    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    # print()
    # print('LEARNING_RATE!!!!', lr)
    # print('NUM_STEPS!!!', num_steps)
    # print('in edit func')
    for i in range(len(probes)):
        for j in range(num_updates):
            probey = probes[i][j]
            gold_label = gold_labels[i][j]
            model_ft = t5_distill(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, lr=lr, 
                        num_steps=num_steps, gold_label=gold_label,
                        max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                        sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search, batch_size=batch_size)
        # print('ONE MINI ITERATION')
            model_raw = model_ft
    model_ft.eval()
    # print('ONE ITERATION IN EDIT FUNC!')


    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    # print('POST EDIT LOGITS')
    # print(post_edit_logits)
    # print(blob)
    # print()

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)



def distill_t5(batch, model_raw, teacher_type, teacher, tokenizer, context, probey, gold_label, device, lr, num_steps, max_length, sample_temperature, top_p, softmax_temperature, 
               repetition_penalty, top_k, length_penalty, beam_search, batch_size,  specificity_batches=None, dataset_name=None):
    # print(batch)
    # print(blob)

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    model_original = copy.deepcopy(model_raw)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits


    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    # print()
    # print('LEARNING_RATE!!!!', lr)
    # print('NUM_STEPS!!!', num_steps)
    # print('in edit func')

    model_ft = t5_distill(model=model_raw, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer,  context=context, probey=probey, device=device, dataset_name=dataset_name, lr=lr, 
                       num_steps=num_steps, gold_label=gold_label, 
                       max_length=max_length, top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                       sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search, batch_size=batch_size)
    model_ft.eval()

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    # print('POST EDIT LOGITS')
    # print(post_edit_logits)
    # print(blob)
    # print()

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)



def curricula_t5(batch, model_raw, context, device, lr, num_steps, specificity_batches=None,
          dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    model_original = copy.deepcopy(model_raw)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    # print('PRE EDIT LOGITS')
    # print(pre_edit_logits)
    # print()
    # print()
    # print('LEARNING_RATE!!!!', lr)
    # print('NUM_STEPS!!!', num_steps)

    model_ft = curricula(model_raw, context=context, device=device, lr=lr, num_steps=num_steps)

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    # print('POST EDIT LOGITS')
    # print(post_edit_logits)

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)



def ft_t5(batch, model_ft, model_raw=None, specificity_batches=None,
          dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_raw, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_raw, model_ft,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

def ft_llama(batch, model_ft, model_raw=None, specificity_batches=None,
           dataset_name=None):

    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_raw, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_raw, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def ft_gpt(batch, model_ft, model_raw=None, specificity_batches=None,
           dataset_name=None):

    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_raw, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_raw, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)
    
def ft_gpt2(batch, model_ft, model_raw=None, specificity_batches=None,
           dataset_name=None):

    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_raw, model_ft, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_raw, model_ft,
                                                  specificity_batches,
                                                  shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)



def prepend_def_t5(batch_pre, batch_post, model):
    # No Def
    with torch.no_grad():
        ex = batch_pre["edit_inner"][0]['probe_sentence']
        ex['labels'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
            0].unsqueeze(0)  # Dummy label
        ex['decoder_attention_mask'] = \
        batch_pre["edit_inner"][0]['labels']['attention_mask'][0].unsqueeze(0)
        pre_edit_logits= model(**ex).logits

    # Prepend def
    with torch.set_grad_enabled(False):
        ex = batch_post["edit_inner"][0]['probe_sentence']
        ex['labels'] = batch_post["edit_inner"][0]['labels']['input_ids'][
            0].unsqueeze(0)  # Dummy label
        ex['decoder_attention_mask'] = \
        batch_post["edit_inner"][0]['labels']['attention_mask'][0].unsqueeze(0)
        post_edit_logits = model(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch_pre['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            pre_label = batch_pre["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, pre_label, shift=False))
            post_label = batch_post["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            post_edit_dict.append(
                get_log_probs(post_edit_logits, post_label, shift=False))

    post_loc_dict = None
    pre_loc_dict = None

    return pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, post_loc_dict, pre_loc_dict


def prepend_def_gpt(batch_pre, batch_post, model):
    # No def
    with torch.no_grad():
        ex = batch_pre["edit_inner"][0]['probe_sentence']
        ex['labels'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
            0].unsqueeze(0)  # Dummy label
        ex['attention_mask'] = batch_pre["edit_inner"][0]['labels'][
            'attention_mask'][0].unsqueeze(0)
        pre_edit_logits = model(**ex).logits

    # Prepend def
    with torch.set_grad_enabled(False):
        ex = batch_post["edit_inner"][0]['probe_sentence']
        ex['labels'] = \
        batch_post["edit_inner"][0]['probe_sentence']['input_ids'][0].unsqueeze(0)
        ex['attention_mask'] = batch_post["edit_inner"][0]['probe_sentence'][
            'attention_mask'][0].unsqueeze(0)
        post_edit_logits = model(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch_pre['edit_inner'][0]['labels']['input_ids'].size(
            0)
        pre_edit_dict = []
        post_edit_dict = []

        for i in range(n_probe_labels):
            pre_label = batch_pre["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, pre_label, shift=False))
            post_label = \
            batch_post["edit_inner"][0]['probe_sentence']['input_ids'][
                i].unsqueeze(0)
            post_edit_dict.append(
                get_log_probs(post_edit_logits, post_label, shift=False))

    post_loc_dict = None
    pre_loc_dict = None

    return pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, post_loc_dict, pre_loc_dict

def prepend_def_llama(batch_pre, batch_post, model):
    # No def
    with torch.no_grad():
        ex = batch_pre["edit_inner"][0]['probe_sentence']
        ex['labels'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
            0].unsqueeze(0)  # Dummy label
        ex['attention_mask'] = batch_pre["edit_inner"][0]['labels'][
            'attention_mask'][0].unsqueeze(0)
        pre_edit_logits = model(**ex).logits

    # Prepend def
    with torch.set_grad_enabled(False):
        ex = batch_post["edit_inner"][0]['probe_sentence']
        ex['labels'] = \
        batch_post["edit_inner"][0]['probe_sentence']['input_ids'][0].unsqueeze(0)
        ex['attention_mask'] = batch_post["edit_inner"][0]['probe_sentence'][
            'attention_mask'][0].unsqueeze(0)
        post_edit_logits = model(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch_pre['edit_inner'][0]['labels']['input_ids'].size(
            0)
        pre_edit_dict = []
        post_edit_dict = []

        for i in range(n_probe_labels):
            pre_label = batch_pre["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, pre_label, shift=False))
            post_label = \
            batch_post["edit_inner"][0]['probe_sentence']['input_ids'][
                i].unsqueeze(0)
            post_edit_dict.append(
                get_log_probs(post_edit_logits, post_label, shift=False))

    post_loc_dict = None
    pre_loc_dict = None

    return pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, post_loc_dict, pre_loc_dict


def mend_gpt(batch, model, specificity_batches=None, dataset_name=None):

    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model(**ex)

    # Edit a model
    edited_model, model_info = model.edit(batch["definition"])  # definition

    with torch.set_grad_enabled(False):
        post_edit_logits = edited_model(**ex)

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(pre_edit_logits, label))
            post_edit_dict.append(get_log_probs(post_edit_logits, label))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None

    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model, edited_model, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model, edited_model,
                                                  specificity_batches,
                                                  shift=True)



    del edited_model

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)
    
def rome_gpt(batch, model, tok, request, specificity_batches=None, dataset_name=None):

    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model(**ex).logits
        # print(request[0]['prompt'])
        # print('in rome_gpt, pre!')
        # print(pre_edit_logits)
    model_original = copy.deepcopy(model)

    # Edit a model
    edited_model, orig_weights = demo_model_editing(model, tok, request, generation_prompts=None, alg_name='ROME') # definition
    # print(model==edited_model)
    with torch.set_grad_enabled(False):
        post_edit_logits = edited_model(**ex).logits
        # print(request[0]['prompt'])
        # print('in rome_gpt, post')
        # print(post_edit_logits)
    #     print(model==edited_model)
    # print(torch.eq(pre_edit_logits, post_edit_logits))
    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(pre_edit_logits, label))
            post_edit_dict.append(get_log_probs(post_edit_logits, label))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None

    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model_original, edited_model, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model_original, edited_model,
                                                  specificity_batches,
                                                  shift=True)



    #del edited_model

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def mend_t5(batch, model, specificity_batches=None, dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)  # Dummy label
    ex['decoder_attention_mask'] = \
        batch["edit_inner"][0]['labels']['attention_mask'][0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model(**ex)

    # Edit a model
    edited_model, model_info = model.edit(batch["definition"])  # definition

    with torch.set_grad_enabled(False):
        post_edit_logits = edited_model(**ex)

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(pre_edit_logits, label))
            post_edit_dict.append(get_log_probs(post_edit_logits, label))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        if dataset_name == 'ecbd':
            pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
                model, edited_model, specificity_batches)
        else:
            pre_loc_dicts, post_loc_dicts = \
            compute_specificity_entity_inferences(model, edited_model,
                                                  specificity_batches,
                                                  shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)

