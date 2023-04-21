import copy
import random
import os
import torch
import types
import yaml
from collections import defaultdict
# from alive_progress import alive_bar
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from .metrics import compute_perplexity_gpt, compute_perplexity_t5, compute_perplexity_llama
from .metrics import compute_dist_over_labels_gpt, compute_dist_over_labels_t5
from .trainer import finetuning, apply_ft_distill_gpt
from .edit_func import ft_gpt, ft_t5, prepend_def_t5, prepend_def_gpt, curricula_t5, ping_t5, teacher_eval, distill_t5, train_distill_t5, vanilla_distill_t5, null_t5, curricula_distill_t5, masked_distill_t5
from .edit_func import mend_gpt, mend_t5, distill_gpt, ft_distill_t5, ft_distill_gpt, multiple_mask_distill_t5, ent_str_distill_gpt
from .edit_func import ft_llama, ft_distill_llama, distill_llama, prepend_def_llama
from .data_utils import to_tsr_gpt_ecbd, to_tsr_t5_ecbd, to_tsr_llama_ecbd, load_json
from .data_utils import format_gpt_data, format_gpt_data_entity_inferences, format_gpt2_data, format_gpt_data_entity_inferences
from .data_utils import to_tsr_gpt_entity_inference, to_tsr_t5_entity_inference
from .data_utils import SPECIFICITY_DATA_PATH
WITHELD_DATA_PATH = '/data/shankar/ping_knowledge_injection/data/ecbd/ecbd_witheld_final.json'
from .pseudo_input import pseudo_input
# import evaluate
# BLEU = evaluate.load('bleu')
# BERT_SCORE = evaluate.load('bertscore')
# from .data_utils import MEND_DIR, MEND_MODEL_DIR, SPECIFICITY_DATA_PATH
# from .mend.mend import MEND
teacher_path = '/data/shankar/ping_pd/output/chat_teacher_dir'
input_generator_path = '/data/shankar/ping_pd/output/chat_input_generator'


def convert_dict_to_namespace(d):
    ns = types.SimpleNamespace()
    for k, v in d.items():
        if 'lr' in k:
            v = float(v)
        setattr(ns, k, v)
    return ns

def split_array(arr, m):
    if m <= 0 or m > len(arr):
        raise ValueError("m should be greater than 0 and less than the length of the input array")
    random.shuffle(arr)

    sub_arrays = [arr[i:i + m] for i in range(0, len(arr), m)]

    return sub_arrays

def run_edit_entity_inference(data,
                              dataset_name,
                              edit_method,
                              device,
                              train_params,
                              model_name=None,
                              random_def=None):

    # Load a raw model and tokenizer.
    # if edit_method == 'teacher_eval':
    #     model_raw = AutoModelForSeq2SeqLM.from_pretrained(teacher_path,)
    #     tokenizer = AutoTokenizer.from_pretrained("t5-large", use_fast=False,)
    #     tokenizer.pad_token = tokenizer.eos_token
    #     to_tsr = to_tsr_t5_entity_inference
    #     # input_generator = AutoModelForSeq2SeqLM.from_pretrained(input_generator_path,)

    if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
        model_raw = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_entity_inference
    elif train_params['BASE_MODEL'] in ['t5-large', 't5-base', 't5-3b']:
        model_raw = T5ForConditionalGeneration.from_pretrained(
            train_params['BASE_MODEL'])
        tokenizer = T5Tokenizer.from_pretrained(train_params['BASE_MODEL'])
        to_tsr = to_tsr_t5_entity_inference
    elif train_params['BASE_MODEL'] == 'llama-7b':
        model_raw = LlamaForCausalLM.from_pretrained('/data/shankar/llama')
        tokenizer = LlamaTokenizer.from_pretrained('/data/shankar/llama')
    else:
        raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
    model_raw = model_raw.to(device)

    all_outputs = []

    edit_func = None
    model_ft = None
    # Select edit function.
    if edit_method == 'ft':  # Finetuned on all examples together.
        assert model_name is not None, 'FT: Finetuned model must be provided.'
        # Load a finetuned model.
        checkpoint = f'/mnt/data1/yasu/newent/ft_outputs/{model_name}/model_files'
        print(model_name)
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
            model_ft = GPTNeoForCausalLM.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
            model_ft = T5ForConditionalGeneration.from_pretrained(checkpoint)
        else:
            raise NotImplementedError(
                'Currently, we use either GPT-Neo or T5.')
        model_ft = model_ft.to(device)
    elif edit_method == 'null':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = null_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = null_t5
    elif edit_method == 'ft_per_ex':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = ft_llama
    elif edit_method == 'curricula':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = curricula_t5
    elif edit_method == 'teacher_eval':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = teacher_eval

    elif edit_method == 'ping':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ping_t5
    elif edit_method == 'distill':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = distill_llama
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
    
    elif edit_method == 'curricula_distill':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = curricula_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
    
    elif edit_method == 'masked_distill':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = curricula_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)

    elif edit_method == 'train_distill':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = train_distill_t5

    elif edit_method == 'vanilla_distill':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = vanilla_distill_t5

    elif edit_method in ['prepend_def', 'prepend_sent', 'random_def',
                         'sanity_check']:
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = prepend_def_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = prepend_def_t5
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = prepend_def_llama
    elif edit_method == 'mend':

        # Mend yaml
        with open(os.path.join(MEND_DIR, 'mend.yaml'), 'r') as f:
            mend_cfg = yaml.safe_load(f)
        _config = convert_dict_to_namespace(mend_cfg)
        _config.mend = convert_dict_to_namespace(mend_cfg['mend'])

        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            with open(os.path.join(MEND_DIR, 'gptneo13.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(
                MEND_MODEL_DIR, 'gpt-neo-1.3B.2022-12-04_12-59-44_5903457054')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_gpt

        elif train_params['BASE_MODEL'] == 't5-large':
            # Model yaml
            with open(os.path.join(MEND_DIR, 't5large_gen.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(MEND_MODEL_DIR,
                                     't5-large.2022-02-12_15-17-56_1732139287')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_t5

    else:
        raise NotImplementedError

    # with alive_bar(total=len(data)) as bar:
    for i, ex in enumerate(data):
        output = {'ex_id': ex['ex_id']}
        label = ex['label']
        batch = to_tsr(tokenizer, ex, device)
        #bleu_score = batch['bleu_score']
        #bert_score = batch['bert_score']
        #bleurt_score = batch['bleurt_score']
        #meteor_score = batch['meteor_score']

        specificity_batches = None
        if train_params['COMPUTE_SPECIFICITY']:
            specificity_data = [ex for j, ex in enumerate(data) if i != j]
            specificity_batches = [
                to_tsr(tokenizer, ex, device) for ex in specificity_data]

        if edit_method == 'ft':  # Finetuned on all examples together.
            _, _, \
            pre_edit_dict, post_edit_dict, \
            post_loc_dict, pre_loc_dict = edit_func(batch,
                                                    model_ft,
                                                    model_raw=model_raw)
        elif edit_method == 'null':
            model_ft = copy.deepcopy(model_raw)
            model_ft = model_ft.to(device) 
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                model_ft,
                model_raw=model_raw,
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)
        elif edit_method == 'ft_per_ex':
            model_ft = copy.deepcopy(model_raw)
            model_ft = model_ft.to(device)
            model_ft, loss = finetuning(model_ft,
                                        tokenizer,
                                        ex,
                                        train_params,
                                        device)  
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                model_ft,
                model_raw=model_raw,
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)



        elif edit_method == 'curricula':
            model_ft = copy.deepcopy(model_raw) 
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                model_raw=model_ft, context=ex['context'], device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], 
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)


        elif edit_method == 'distill':
            model_ft = copy.deepcopy(model_raw) 
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, context=ex['context'], probey=ex['probey'], device=device,lr=train_params['LEARNING_RATE'], 
                num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                repetition_penalty=train_params['REPETITION_PENALTY'], top_k=train_params['TOP_K'], length_penalty=train_params['LENGTH_PENALTY'], 
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)

        
        elif edit_method == 'curricula_distill':
            model_ft = copy.deepcopy(model_raw)
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                    batch,
                    model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'],teacher=teacher, tokenizer=tokenizer, 
                    example=ex, device=device, initial_noise=train_params['INITIAL_NOISE'], final_noise=train_params['FINAL_NOISE'],
                    lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                    repetition_penalty=train_params['REPETITION_PENALTY'], top_k=train_params['TOP_K'], length_penalty=train_params['LENGTH_PENALTY'], 
                    specificity_batches=specificity_batches,
                    dataset_name=dataset_name)
    
    
        
        elif edit_method == 'vanilla_distill':
            model_ft = copy.deepcopy(model_raw)  
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                model_raw=model_ft, context=ex['context'], probey=ex['probey'], device=device,lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                repetition_penalty=train_params['REPETITION_PENALTY'], top_k=train_params['TOP_K'], length_penalty=train_params['LENGTH_PENALTY'], 
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)




        elif edit_method == 'prepend_def':
            batch_prepended_def = to_tsr(tokenizer,
                                         ex,
                                         device,
                                         prepend_def=True,
                                         prepend_sent=False,
                                         random_def=None)
            # print('HERE IS UN-ENCODED BATCH')
            # # print(batch['edit_inner'][0]['probe_sentence'])
            # print(tokenizer.decode(batch['edit_inner'][0]['probe_sentence']['input_ids'][0], skip_special_tokens=True))
            # print('HERE IS UN-ENCODED NEW BATCH')
            print(tokenizer.decode(batch_prepended_def['edit_inner'][0]['probe_sentence']['input_ids'][0], skip_special_tokens=True))
            _, _, \
            pre_edit_dict, post_edit_dict, \
            post_loc_dict, pre_loc_dict = edit_func(batch,
                                                    batch_prepended_def,
                                                    model_raw)
        elif edit_method == 'random_def':
            batch_prepended_def = to_tsr(tokenizer,
                                         ex,
                                         device,
                                         prepend_def=False,
                                         prepend_sent=False,
                                         random_def=random_def)
            _, _, \
            pre_edit_dict, post_edit_dict, \
            post_loc_dict, pre_loc_dict = edit_func(batch,
                                                    batch_prepended_def,
                                                    model_raw)
        elif edit_method == 'sanity_check':
            model_ft = copy.deepcopy(model_raw)
            model_ft = model_ft.to(device)
            model_ft, loss = finetuning(model_ft,
                                        tokenizer,
                                        ex,
                                        train_params,
                                        device)
            _, _, \
            pre_edit_dict, post_edit_dict, \
            post_loc_dict, pre_loc_dict = edit_func(batch,
                                                    model_ft,
                                                    model_raw=model_raw)
        elif edit_method == 'mend':
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                mend_model,
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)

        else:
            raise

        assert len(batch["edit_inner"]) == 1, len(batch["edit_inner"])

        j = 0
        # Assuming only 1 probe sentence.
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':

            labels, pre_probs, pre_lls = compute_dist_over_labels_gpt(
                tokenizer,
                pre_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels'],
                batch["edit_inner"][j]['left_context_ps'],
                batch["edit_inner"][j]['right_context_ps']
            )

            labels, post_probs, post_lls = compute_dist_over_labels_gpt(
                tokenizer,
                post_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels'],
                batch["edit_inner"][j]['left_context_ps'],
                batch["edit_inner"][j]['right_context_ps']
            )

            # Release GPU memory.
            pre_edit_dict = None
            post_edit_dict = None

            results_specificity = None
            if train_params['COMPUTE_SPECIFICITY']:
                results_specificity = []
                assert len(specificity_batches) == len(pre_loc_dicts) \
                       == len(post_loc_dicts)
                for k in range(len(specificity_batches)):

                    s_batch = specificity_batches[k]
                    s_labels, s_pre_probs, s_pre_lls = \
                    compute_dist_over_labels_gpt(
                        tokenizer,
                        pre_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels'],
                        s_batch["edit_inner"][0]['left_context_ps'],
                        s_batch["edit_inner"][0]['right_context_ps']
                    )

                    s_labels, s_post_probs, s_post_lls = \
                    compute_dist_over_labels_gpt(
                        tokenizer,
                        post_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels'],
                        s_batch["edit_inner"][0]['left_context_ps'],
                        s_batch["edit_inner"][0]['right_context_ps']
                    )
                    s_label = specificity_data[k]['label']
                    s_result = [p for p in
                              zip(s_labels, s_pre_lls, s_post_lls,
                                  s_pre_probs, s_post_probs)
                              if p[0] == s_label][0]
                    s_pred_dist = [
                        list(zip(s_labels, s_pre_lls, s_post_lls,
                                 s_pre_probs, s_post_probs)), s_label]
                    results_specificity.append(
                        {'results': s_result, 'probs': s_pred_dist})

            pre_loc_dicts = None
            post_loc_dicts = None

        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:

            labels, pre_probs, pre_lls = compute_dist_over_labels_t5(
                tokenizer,
                pre_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels']
            )

            labels, post_probs, post_lls = compute_dist_over_labels_t5(
                tokenizer,
                post_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels']
            )
            # Release GPU memory.
            pre_edit_dict = None
            post_edit_dict = None

            results_specificity = None
            if train_params['COMPUTE_SPECIFICITY']:
                results_specificity = []
                assert len(specificity_batches) == len(pre_loc_dicts) \
                       == len(post_loc_dicts)
                for k in range(len(specificity_batches)):

                    s_batch = specificity_batches[k]
                    s_labels, s_pre_probs, s_pre_lls = \
                    compute_dist_over_labels_t5(
                        tokenizer,
                        pre_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels']
                    )

                    s_labels, s_post_probs, s_post_lls = \
                    compute_dist_over_labels_t5(
                        tokenizer,
                        post_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels']
                    )

                    s_label = specificity_data[k]['label']
                    s_result = [p for p in
                              zip(s_labels, s_pre_lls, s_post_lls,
                                  s_pre_probs, s_post_probs)
                              if p[0] == s_label][0]
                    s_pred_dist = [
                        list(zip(s_labels, s_pre_lls, s_post_lls,
                                 s_pre_probs, s_post_probs)), s_label]
                    results_specificity.append(
                        {'results': s_result, 'probs': s_pred_dist})

            pre_loc_dicts = None
            post_loc_dicts = None

        else:
            raise NotImplementedError

        result = None
        pred_dist = None
        if label in labels:
            result = [p for p in
                 zip(labels, pre_lls, post_lls, pre_probs, post_probs)
                 if p[0] == label][0]
            pred_dist = [list(zip(labels, pre_lls, post_lls, pre_probs,
                          post_probs)), label]
        elif isinstance(label, list):
            label_scores = []
            all_scores = []
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                all_scores.append(p)
                if p[0] in label:
                    label_scores.append(p)
            result = label_scores
            pred_dist = [all_scores, label]
        else:
            print('-' * 60)
            print('Probe Sentence {}: {}'.format(j,
                                                 ex['probe_sentences'][
                                                     f'template_{j}'][
                                                     'probe_sentence']))
            print('WARNING: Label not found! {}'.format(label))
            print('         Labels {}'.format(labels))
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                print(p)

        # assert len(results_specificity) == len(data) - 1, \
        #     (len(results_specificity), len(data))

        output['results'] = result
        output['probs'] = pred_dist
        # output['sim_scores'] = {
        #     'bleu_score': bleu_score,
        #     'bert_score': bert_score,
        # }
        output['specificity'] = results_specificity
        all_outputs.append(output)
        # bar()

    return all_outputs


def run_edit_ecbd(data,
                  dataset_name,
                  edit_method,
                  device,
                  train_params,
                  model_name=None,
                  random_def=None,
                  oracle_ft=False,
                  specificity_data=None,
                  witheld_data=None):

    # Load a raw model and tokenizer.
    if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
        model_raw = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_ecbd
        # print('model set stuff done')
    elif train_params['BASE_MODEL'] in ['t5-large', 't5-base', 't5-3b']:
        model_raw = T5ForConditionalGeneration.from_pretrained(
            train_params['BASE_MODEL'])
        tokenizer = T5Tokenizer.from_pretrained(train_params['BASE_MODEL'])
        to_tsr = to_tsr_t5_ecbd
    elif train_params['BASE_MODEL'] == 'gpt2-xl':
        model_raw = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        tok = AutoTokenizer.from_pretrained('gpt2-xl')
        tokenizer.pad_token = tokenizer.eos_token
        tok.pad_token = tok.eos_token
        to_tsr = to_tsr_gpt_ecbd
    elif train_params['BASE_MODEL'] == 'llama-7b':
        model_raw = LlamaForCausalLM.from_pretrained('/data/shankar/llama')
        tokenizer = LlamaTokenizer.from_pretrained('/data/shankar/llama')
        to_tsr = to_tsr_llama_ecbd
    else:
        raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
    # print('before device!')
    model_raw = model_raw.to(device)
    # print('done with setting to device!')

    # Finetuned model.
    model_ft = None

    # Select edit function.
    if edit_method == 'ft':
        assert model_name is not None, 'FT: Finetuned model must be provided.'
        # Load a finetuned model.
        checkpoint = f'/mnt/data1/yasu/newent/ft_outputs/{model_name}/model_files'
        # print(model_name)
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
            model_ft = GPTNeoForCausalLM.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
            model_ft = T5ForConditionalGeneration.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] == 'llama-7b':
            edit_func = ft_llama
        elif train_params['BASE_MODEL'] == 'gpt2-xl':
            edit_func = ft_gpt
            model_ft = GPT2LMHeadModel.from_pretrained(checkpoint)
        else:
            raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
        model_ft = model_ft.to(device)
    elif edit_method == 'null':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = null_t5
    elif edit_method == 'ft_per_ex':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = ft_llama
    elif edit_method == 'curricula':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = curricula_t5
    elif edit_method == 'ent_str_distill' or edit_method == 'null_str_distill':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ent_str_distill_gpt
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                teacher = GPTNeoForCausalLM.from_pretrained(train_params['TEACHER_MODEL'])
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                teacher = GPT2LMHeadModel.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = ent_str_distill_llama
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
    elif edit_method == 'distill':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = distill_gpt
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                teacher = GPTNeoForCausalLM.from_pretrained(train_params['TEACHER_MODEL'])
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                teacher = GPT2LMHeadModel.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = distill_llama
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)

    elif edit_method == 'ft_distill':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_distill_gpt
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                teacher = GPTNeoForCausalLM.from_pretrained(train_params['TEACHER_MODEL'])
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                teacher = GPT2LMHeadModel.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = ft_distill_llama
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
    
    elif edit_method == 'ft_distill_multiple':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                teacher = GPTNeoForCausalLM.from_pretrained(train_params['TEACHER_MODEL'])
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                teacher = GPT2LMHeadModel.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = ft_llama
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)

    elif edit_method == 'random_distill':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = random_distill_gpt
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                teacher = GPTNeoForCausalLM.from_pretrained(train_params['TEACHER_MODEL'])
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                teacher = GPT2LMHeadModel.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
    
    elif edit_method == 't5_multiple_mask_distill':
        if train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = multiple_mask_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)

    elif edit_method == 'vanilla_distill':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = vanilla_distill_gpt
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                "gpt-neo-1.3B", 
                use_fast=False,  
            )
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = vanilla_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                "t5-3b", 
                use_fast=False,  
            )

    elif edit_method in ['prepend_def', 'prepend_sent', 'random_def',
                         'sanity_check']:
        print('in prepend def!')
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = prepend_def_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = prepend_def_t5
        elif train_params['BASE_MODEL'] in ['llama-7b']:
            edit_func = prepend_def_llama
    elif edit_method == 'mend':

        # Mend yaml
        with open(os.path.join(MEND_DIR, 'mend.yaml'), 'r') as f:
            mend_cfg = yaml.safe_load(f)
        _config = convert_dict_to_namespace(mend_cfg)
        _config.mend = convert_dict_to_namespace(mend_cfg['mend'])

        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            with open(os.path.join(MEND_DIR, 'gptneo13.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(
                MEND_MODEL_DIR, 'gpt-neo-1.3B.2022-12-04_12-59-44_5903457054')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_gpt

        elif train_params['BASE_MODEL'] == 't5-large':

            # Model yaml
            with open(os.path.join(MEND_DIR, 't5large_gen.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)

            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(MEND_MODEL_DIR, 't5-large.2022-02-12_15-17-56_1732139287')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_t5

    else:
        raise NotImplementedError

    if specificity_data is not None:
        # print('IN HERE!')
        specificity_batches = [
            to_tsr(tokenizer, _ex, device) for _ex in
            specificity_data]
        # print(specificity_batches)
    all_outputs = []
    if 'WITHELD' in train_params.keys() and edit_method == 'ft_distill_multiple':
        example_set = [ent[1] for i, ent in enumerate(data[:])]
        witheld_set = [ent for i, ent in enumerate(witheld_data[:])]
        random.shuffle(witheld_set)
        standardized_set_for_distillation = witheld_set[:train_params['NUM_BEFORE']+train_params['NUM_AFTER']] #take first n elements to use for distillation 
        distillation_set_before = standardized_set_for_distillation[:train_params['NUM_BEFORE']] #set of stuff to apply distillation to before entity 
        distillation_set_after = standardized_set_for_distillation[train_params['NUM_BEFORE']:train_params['NUM_BEFORE']+train_params['NUM_AFTER']] #set of stuff to apply distillation to after entity
        model_ft = copy.deepcopy(model_raw)
        if train_params['NUM_BEFORE']!=0:
            contexts = []
            unmasked_probe_set = []
            ent_strs = []
            for ex in distillation_set_before:
                contexts.append(ex['context'])
                unmasked_probe_set.append(ex['augmented_probes'])
                ent_strs.append(ex['ent_str'])
            model_ft = apply_ft_distill_gpt(model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, contexts=contexts, 
                                            unmasked_probe_set=unmasked_probe_set, ent_strs=ent_strs, device=device, lr=train_params['LEARNING_RATE'], 
                                            num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'], 
                                            repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'],
                                            length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], 
                                            batch_size=train_params['DISTILL_BATCH_SIZE'], specificity_batches=specificity_batches, num_probes=train_params['NUM_PROBES'], 
                                            num_updates=train_params['NUM_UPDATES'], dataset_name=dataset_name)              
        for examples in example_set:
            ex = examples[0]
            model_ft = apply_ft_distill_gpt(model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, contexts=[ex['context']], 
                                            unmasked_probe_set=[ex['augmented_probes']], ent_strs=[ex['ent_str']],device=device, lr=train_params['LEARNING_RATE'], 
                                            num_steps=train_params['TRAIN_EPOCHS'],  max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'], 
                                            repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'],
                                            length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], 
                                            batch_size=train_params['DISTILL_BATCH_SIZE'], specificity_batches=specificity_batches, num_probes=train_params['NUM_PROBES'], 
                                            num_updates=train_params['NUM_UPDATES'], dataset_name=dataset_name)   
            if train_params['NUM_AFTER']!=0:
                contexts = []
                unmasked_probe_set = []
                ent_strs = []
                for ex in distillation_set_after:
                    ex = examples[0]
                    contexts.append(ex['context'])
                    unmasked_probe_set.append(ex['augmented_probes'])
                    ent_strs.append(ex['ent_str'])
                model_ft = apply_ft_distill_gpt(model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, contexts=contexts, 
                                                unmasked_probe_set=unmasked_probe_set, ent_strs=ent_strs, device=device, lr=train_params['LEARNING_RATE'], 
                                                num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'], 
                                                repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'],
                                                length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], 
                                                batch_size=train_params['DISTILL_BATCH_SIZE'], specificity_batches=specificity_batches, num_probes=train_params['NUM_PROBES'], 
                                                num_updates=train_params['NUM_UPDATES'], dataset_name=dataset_name) 
            for ex in examples[:]:
                    output = {'ex_id': ex['ex_id']}
                    batch = to_tsr(tokenizer, ex, device)
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch=batch,
                        model_ft=model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                    j = 0
                    if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
                        if edit_method == 'prepend_def':
                            pre_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                pre_edit_logits,
                                batch["edit_inner"][j]['probe_sentence'][
                                    'input_ids'],
                                batch["edit_inner"][j]['probe_sentence'][
                                    'attention_mask'],
                                batch["edit_inner"][j]['probe_sentence'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            post_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                post_edit_logits,
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence']['input_ids'],
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence']['attention_mask'],
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence'],
                                batch_prepended_def["edit_inner"][j][
                                    'left_context_ps'],
                                batch_prepended_def["edit_inner"][j][
                                    'right_context_ps']
                            )
                            pre_edit_logits = None
                            post_edit_logits = None
                            results_specificity = None
                            # print('fourth place')
                            # print(type(specificity_batches))
                            if train_params['COMPUTE_SPECIFICITY']:
                                results_specificity = []
                                # print(type(specificity_batches))
                                # print(type(pre_edit_dict))
                                # print(type(post_edit_dict))
                                # print(len(specificity_batches))
                                # print(len(pre_edit_dict))
                                # print(len(post_edit_dict))
                                assert len(specificity_batches) == len(
                                    pre_edit_dict) \
                                    == len(post_edit_dict)
                                for k in range(len(specificity_batches)):
                                    s_batch = specificity_batches[k]
                                    # print(specificity_data[k]['probe_sentences'][
                                    #                            'template_0']['label'][13:-13])
                                    s_pre_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        pre_edit_dict[k],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['probe_sentence'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    s_post_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        post_edit_dict[k],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['probe_sentence'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    results_specificity.append(
                                        {'pre': s_pre_perp_loss[0],
                                        'post': s_post_perp_loss[0]})
                            # print(pre_perp_loss)
                            # print(post_perp_loss)

                        else:
                            pre_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                pre_edit_logits,
                                batch["edit_inner"][j]['labels']['input_ids'],
                                batch["edit_inner"][j]['labels']['attention_mask'],
                                batch["edit_inner"][j]['labels'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            post_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                post_edit_logits,
                                batch["edit_inner"][j]['labels']['input_ids'],
                                batch["edit_inner"][j]['labels']['attention_mask'],
                                batch["edit_inner"][j]['labels'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            pre_edit_logits = None
                            post_edit_logits = None

                            results_specificity = None
                            if train_params['COMPUTE_SPECIFICITY']:
                                results_specificity = []
                                assert len(specificity_batches) == len(
                                    pre_loc_logits) \
                                    == len(post_loc_logits)
                                for k in range(len(specificity_batches)):
                                    s_batch = specificity_batches[k]
                                    # print(specificity_data[k]['probe_sentences'][
                                    #                            'template_0']['label'][13:-13])
                                    s_pre_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        pre_loc_logits[k],
                                        s_batch["edit_inner"][0]['labels'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['labels'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['labels'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    s_post_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        post_loc_logits[k],
                                        s_batch["edit_inner"][0]['labels'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['labels'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['labels'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    results_specificity.append(
                                        {'pre': s_pre_perp_loss[0],
                                        'post': s_post_perp_loss[0]})

                        pre_loc_logits = None
                        post_loc_logits = None

                    elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
                        label_ids = batch["edit_inner"][0]['labels']['input_ids']
                        label_attention_mask = batch["edit_inner"][0]['labels'][
                            'attention_mask']
                        pre_perp_loss = compute_perplexity_t5(tokenizer,
                                                            pre_edit_logits,
                                                            label_ids,
                                                            label_attention_mask)
                        post_perp_loss = compute_perplexity_t5(tokenizer,
                                                            post_edit_logits,
                                                            label_ids,
                                                            label_attention_mask)

                        pre_edit_logits = None
                        post_edit_logits = None

                        results_specificity = None
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(pre_loc_logits) \
                                == len(post_loc_logits)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                s_pre_perp_loss = compute_perplexity_t5(
                                    tokenizer,
                                    pre_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask']
                                )

                                s_post_perp_loss = compute_perplexity_t5(
                                    tokenizer,
                                    post_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})

                        pre_loc_logits = None
                        post_loc_logits = None

                    else:
                        raise NotImplementedError

                    output['pre'] = pre_perp_loss[0]
                    output['post'] = post_perp_loss[0]
                    # output['sim_scores'] = {
                    #     'bleu_score': bleu_score,
                    #     'bert_score': bert_score,
                    #     'bleurt_score': bleurt_score,
                    #     'meteor_score': meteor_score
                    # }
                    output['specificity'] = results_specificity
                    all_outputs.append(output)



    elif edit_method == 'ft_distill_multiple':
        num_per_batch = train_params['NUM_EDITS']
        example_set = []
        example_set = [ent[1] for i, ent in enumerate(data[:])]
        edit_sets = split_array(example_set, num_per_batch)


        for set in edit_sets:
            contexts = []
            unmasked_probe_set = []
            ent_strs = []
            # batches = []
            for examples in set:
                ex = examples[0] #Take the first one
                contexts.append(ex['context'])
                unmasked_probe_set.append(ex['augmented_probes'])
                ent_strs.append(ex['ent_str'])
                # batches.append(to_tsr(tokenizer, ex, device))
            # print(len(contexts))
            # assert len(contexts) == num_per_batch
            
            model_ft = copy.deepcopy(model_raw)
            model_ft = apply_ft_distill_gpt(model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, contexts=contexts, 
                                            unmasked_probe_set=unmasked_probe_set, ent_strs=ent_strs, device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], 
                                            max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'], repetition_penalty=train_params['REPETITION_PENALTY'], 
                                            sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'], length_penalty=train_params['LENGTH_PENALTY'], 
                                            beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], batch_size=train_params['DISTILL_BATCH_SIZE'], 
                                            specificity_batches=specificity_batches, num_probes=train_params['NUM_PROBES'], num_updates=train_params['NUM_UPDATES'], dataset_name=dataset_name)
            
            for i in range(len(set)): 
                examples = set[i]
                # batch = batches[i]
                for ex in examples[:]:
                    output = {'ex_id': ex['ex_id']}
                    batch = to_tsr(tokenizer, ex, device)


                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch=batch,
                        model_ft=model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                    j = 0
                    if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
                        if edit_method == 'prepend_def':
                            pre_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                pre_edit_logits,
                                batch["edit_inner"][j]['probe_sentence'][
                                    'input_ids'],
                                batch["edit_inner"][j]['probe_sentence'][
                                    'attention_mask'],
                                batch["edit_inner"][j]['probe_sentence'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            post_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                post_edit_logits,
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence']['input_ids'],
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence']['attention_mask'],
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence'],
                                batch_prepended_def["edit_inner"][j][
                                    'left_context_ps'],
                                batch_prepended_def["edit_inner"][j][
                                    'right_context_ps']
                            )
                            pre_edit_logits = None
                            post_edit_logits = None
                            results_specificity = None
                            # print('fourth place')
                            # print(type(specificity_batches))
                            if train_params['COMPUTE_SPECIFICITY']:
                                results_specificity = []
                                # print(type(specificity_batches))
                                # print(type(pre_edit_dict))
                                # print(type(post_edit_dict))
                                # print(len(specificity_batches))
                                # print(len(pre_edit_dict))
                                # print(len(post_edit_dict))
                                assert len(specificity_batches) == len(
                                    pre_edit_dict) \
                                    == len(post_edit_dict)
                                for k in range(len(specificity_batches)):
                                    s_batch = specificity_batches[k]
                                    # print(specificity_data[k]['probe_sentences'][
                                    #                            'template_0']['label'][13:-13])
                                    s_pre_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        pre_edit_dict[k],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['probe_sentence'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    s_post_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        post_edit_dict[k],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['probe_sentence'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    results_specificity.append(
                                        {'pre': s_pre_perp_loss[0],
                                        'post': s_post_perp_loss[0]})
                            # print(pre_perp_loss)
                            # print(post_perp_loss)

                        else:
                            pre_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                pre_edit_logits,
                                batch["edit_inner"][j]['labels']['input_ids'],
                                batch["edit_inner"][j]['labels']['attention_mask'],
                                batch["edit_inner"][j]['labels'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            post_perp_loss = compute_perplexity_gpt(
                                tokenizer,
                                post_edit_logits,
                                batch["edit_inner"][j]['labels']['input_ids'],
                                batch["edit_inner"][j]['labels']['attention_mask'],
                                batch["edit_inner"][j]['labels'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            pre_edit_logits = None
                            post_edit_logits = None

                            results_specificity = None
                            if train_params['COMPUTE_SPECIFICITY']:
                                results_specificity = []
                                assert len(specificity_batches) == len(
                                    pre_loc_logits) \
                                    == len(post_loc_logits)
                                for k in range(len(specificity_batches)):
                                    s_batch = specificity_batches[k]
                                    # print(specificity_data[k]['probe_sentences'][
                                    #                            'template_0']['label'][13:-13])
                                    s_pre_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        pre_loc_logits[k],
                                        s_batch["edit_inner"][0]['labels'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['labels'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['labels'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    s_post_perp_loss = compute_perplexity_gpt(
                                        tokenizer,
                                        post_loc_logits[k],
                                        s_batch["edit_inner"][0]['labels'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['labels'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['labels'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    results_specificity.append(
                                        {'pre': s_pre_perp_loss[0],
                                        'post': s_post_perp_loss[0]})

                        pre_loc_logits = None
                        post_loc_logits = None

                    elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
                        label_ids = batch["edit_inner"][0]['labels']['input_ids']
                        label_attention_mask = batch["edit_inner"][0]['labels'][
                            'attention_mask']
                        pre_perp_loss = compute_perplexity_t5(tokenizer,
                                                            pre_edit_logits,
                                                            label_ids,
                                                            label_attention_mask)
                        post_perp_loss = compute_perplexity_t5(tokenizer,
                                                            post_edit_logits,
                                                            label_ids,
                                                            label_attention_mask)

                        pre_edit_logits = None
                        post_edit_logits = None

                        results_specificity = None
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(pre_loc_logits) \
                                == len(post_loc_logits)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                s_pre_perp_loss = compute_perplexity_t5(
                                    tokenizer,
                                    pre_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask']
                                )

                                s_post_perp_loss = compute_perplexity_t5(
                                    tokenizer,
                                    post_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})

                        pre_loc_logits = None
                        post_loc_logits = None

                    else:
                        raise NotImplementedError

                    output['pre'] = pre_perp_loss[0]
                    output['post'] = post_perp_loss[0]
                    # output['sim_scores'] = {
                    #     'bleu_score': bleu_score,
                    #     'bert_score': bert_score,
                    #     'bleurt_score': bleurt_score,
                    #     'meteor_score': meteor_score
                    # }
                    output['specificity'] = results_specificity
                    all_outputs.append(output)
                    # print()
                    # print(output)
                    # print()

    # with alive_bar(total=len(data)) as bar:
    else:
        for i, ent in enumerate(data[:]):
            # print('first place')
            # print(type(specificity_batches))

            # ex contains multiple probe sentences.
            ent_id, examples = ent
            ex_for_finetuning = examples[0]  # Take the first one
            # print(ex_for_finetuning)
            # print(blob)

            if oracle_ft:
                random.shuffle(examples)
                n_ex = len(examples) // 2
                if n_ex:
                    ex_for_training = examples[:n_ex]
                    ex_for_testing = examples[n_ex:]
                    ex_for_finetuning = ex_for_training[0]  # Dummy
                    ex_for_finetuning['definition'] = \
                    ex_for_finetuning['probe_sentences']['template_0'][
                        'probe_sentence']
                    ex_for_finetuning['def_target'] = \
                    ex_for_finetuning['probe_sentences']['template_0']['label']
                    ex_for_finetuning['additional_sentences'] = []
                    for _ex in ex_for_training[1:]:
                        ex_for_finetuning['additional_sentences'].append(
                            [
                                _ex['probe_sentences']['template_0'][
                                    'probe_sentence'],
                                _ex['probe_sentences']['template_0']['label']
                            ]
                        )
                    examples = ex_for_testing
                else:
                    continue

            if edit_method == 'ft_per_ex':
                # Finetune a model
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
                model_ft, loss = finetuning(model_ft,
                                            tokenizer,
                                            ex_for_finetuning,
                                            train_params,
                                            device)  # single instance
            elif edit_method in ['prepend_def', 'random_def']:
                # print('in for loop!')
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
                # print('second place')
                # print(type(specificity_batches))
            elif edit_method == 'sanity_check':
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
                model_ft, loss = finetuning(model_ft, tokenizer, ex_for_finetuning,
                                            train_params, device)
            elif edit_method == 'mend':
                pass
            # else:
            #     raise
            for ex in examples[:]:
                output = {'ex_id': ex['ex_id']}
                batch = to_tsr(tokenizer, ex, device)
                # bleu_score = batch['bleu_score']
                # bert_score = batch['bert_score']
                # bleurt_score = batch['bleurt_score']
                # meteor_score = batch['meteor_score']
                if edit_method == 'null':
                # Finetune a model
                    model_ft = copy.deepcopy(model_raw)
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)

                if edit_method == 'ft_per_ex':
                    print('in here!')
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                elif edit_method == 'prepend_def':
                    print('in pre_edit_logit part!')
                    batch_prepended_def = to_tsr(tokenizer,
                                                ex,
                                                device,
                                                prepend_def=True,
                                                prepend_sent=False,
                                                random_def=None)
                    pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, _, _,  = edit_func(
                        batch,
                        batch_prepended_def,
                        model_ft)
                    # print(type(pre_loc_dict))
                    # print(type(post_loc_dict))
                    # print('third place')
                    # print(type(specificity_batches))
                    # print(pre_edit_logits, post_edit_logits)

                elif edit_method == 'curricula':
                    model_ft = copy.deepcopy(model_raw)
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _  = edit_func(
                        batch,
                        model_raw=model_ft, context=ex['context'], device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], 
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)


                elif edit_method == 'random_def':
                    batch_prepended_def = to_tsr(tokenizer,
                                                ex,
                                                device,
                                                prepend_def=False,
                                                prepend_sent=False,
                                                random_def=random_def)
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    post_loc_dict, pre_loc_dict = edit_func(batch,
                                                            batch_prepended_def,
                                                            model_ft)
                
                elif edit_method == 'null_str_distill': #same as regular train-on-test distillation, but substitute entity string in place of augmented probe
                    model_ft = copy.deepcopy(model_raw) 
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, context=ex['context'], probey='', 
                        device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'], gold_label=ex['label'],
                        length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], batch_size=train_params['DISTILL_BATCH_SIZE'],
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name, ent_str_only=True)
                
                elif edit_method == 'ent_str_distill': #same as regular train-on-test distillation, but substitute entity string in place of augmented probe
                    model_ft = copy.deepcopy(model_raw) 
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, context=ex['context'], probey=ex['ent_str'], 
                        device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'], gold_label=ex['label'],
                        length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], batch_size=train_params['DISTILL_BATCH_SIZE'],
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name, ent_str_only=True)

                elif edit_method == 'distill':
                    # print('in run edit logits calc')
                    model_ft = copy.deepcopy(model_raw) 
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, context=ex['context'], probey=ex['pseudo'], 
                        device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'], gold_label=ex['label'],
                        length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], batch_size=train_params['DISTILL_BATCH_SIZE'],
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                
                elif edit_method == 'ft_distill':
                    # print('in run edit logits calc')
                    model_ft = copy.deepcopy(model_raw)
                    # if 'AFTER_ENT_SPAN' in train_params.keys():
                    #     after_ent_span=True
                    # else:

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, context=ex['context'], probes=ex['masked_augmentations'], ent_str=ex['ent_str'],
                        device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'], gold_labels=ex['augment_labels'],
                        length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], batch_size=train_params['DISTILL_BATCH_SIZE'],
                        specificity_batches=specificity_batches, num_probes=train_params['NUM_PROBES'], unmasked_probes=ex['augmented_probes'], num_updates=train_params['NUM_UPDATES'],
                        dataset_name=dataset_name)
                    
                elif edit_method == 'random_distill':
                    # print('in run edit logits calc')
                    model_ft = copy.deepcopy(model_raw)
                    # if 'AFTER_ENT_SPAN' in train_params.keys():
                    #     after_ent_span=True
                    # else:

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, context=ex['context'], ent_str=ex['ent_str'],
                        device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'], gold_labels=ex['augment_labels'],
                        length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], batch_size=train_params['DISTILL_BATCH_SIZE'],
                        specificity_batches=specificity_batches, num_probes=train_params['NUM_PROBES'], num_updates=train_params['NUM_UPDATES'], unmasked_probes=ex['augmented_probes'],
                        dataset_name=dataset_name)
                
                elif edit_method == 't5_multiple_mask_distill':
                    # print('in run edit logits calc')
                    model_ft = copy.deepcopy(model_raw)

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, context=ex['context'], probes=ex['masked_augmentations'],
                        device=device, lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], sample_temperature=train_params['SAMPLE_TEMPERATURE'], top_k=train_params['TOP_K'], gold_labels=ex['augment_labels'],
                        length_penalty=train_params['LENGTH_PENALTY'], beam_search=train_params['BEAM_SEARCH'], softmax_temperature=train_params['SOFTMAX_TEMP'], 
                        batch_size=train_params['DISTILL_BATCH_SIZE'], specificity_batches=specificity_batches, num_probes=train_params['NUM_PROBES'], 
                        unmasked_probes=ex['augmented_probes'], num_updates=train_params['NUM_UPDATES'],
                        dataset_name=dataset_name)
                
                elif edit_method == 'curricula_distill':
                    model_ft = copy.deepcopy(model_raw)
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher_type=train_params['TEACHER_MODEL'], teacher=teacher, tokenizer=tokenizer, example=ex, device=device, initial_noise=train_params['INITIAL_NOISE'], final_noise=train_params['FINAL_NOISE'],
                        lr=train_params['LEARNING_RATE'], num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], top_k=train_params['TOP_K'], length_penalty=train_params['LENGTH_PENALTY'], 
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                
                
                elif edit_method == 'vanilla_distill':
                    model_ft = copy.deepcopy(model_raw)

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_ft, teacher=teacher, student_tokenizer=tokenizer, teacher_tokenizer=teacher_tokenizer, probey=ex['probey'], device=device,  lr=train_params['LEARNING_RATE'], 
                        num_steps=train_params['TRAIN_EPOCHS'], max_length=train_params['MAX_TARGET_TEXT_LENGTH'], top_p=train_params['TOP_P'],
                        repetition_penalty=train_params['REPETITION_PENALTY'], top_k=train_params['TOP_K'], length_penalty=train_params['LENGTH_PENALTY'], 
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)

                elif edit_method == 'sanity_check':
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    post_loc_dict, pre_loc_dict = edit_func(batch,
                                                            model_ft,
                                                            model_raw=model_raw)
                elif edit_method == 'mend':
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        mend_model,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                # else:
                #     raise

                assert len(batch["edit_inner"]) == 1, len(batch["edit_inner"])

                j = 0
                # Assuming only 1 probe sentence.
                # print('before stuff')
                # print(train_params['BASE_MODEL'])
                if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
                    # print('MADE IT IN HERE!')

                    if edit_method == 'prepend_def':
                        pre_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['probe_sentence'][
                                'input_ids'],
                            batch["edit_inner"][j]['probe_sentence'][
                                'attention_mask'],
                            batch["edit_inner"][j]['probe_sentence'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        post_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            post_edit_logits,
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['input_ids'],
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['attention_mask'],
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence'],
                            batch_prepended_def["edit_inner"][j][
                                'left_context_ps'],
                            batch_prepended_def["edit_inner"][j][
                                'right_context_ps']
                        )
                        pre_edit_logits = None
                        post_edit_logits = None
                        results_specificity = None
                        # print('fourth place')
                        # print(type(specificity_batches))
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            # print(type(specificity_batches))
                            # print(type(pre_edit_dict))
                            # print(type(post_edit_dict))
                            # print(len(specificity_batches))
                            # print(len(pre_edit_dict))
                            # print(len(post_edit_dict))
                            assert len(specificity_batches) == len(
                                pre_edit_dict) \
                                == len(post_edit_dict)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                # print(specificity_data[k]['probe_sentences'][
                                #                            'template_0']['label'][13:-13])
                                s_pre_perp_loss = compute_perplexity_gpt(
                                    tokenizer,
                                    pre_edit_dict[k],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['probe_sentence'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                s_post_perp_loss = compute_perplexity_gpt(
                                    tokenizer,
                                    post_edit_dict[k],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['probe_sentence'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})
                        # print(pre_perp_loss)
                        # print(post_perp_loss)

                    else:
                        pre_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['labels']['input_ids'],
                            batch["edit_inner"][j]['labels']['attention_mask'],
                            batch["edit_inner"][j]['labels'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        post_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            post_edit_logits,
                            batch["edit_inner"][j]['labels']['input_ids'],
                            batch["edit_inner"][j]['labels']['attention_mask'],
                            batch["edit_inner"][j]['labels'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        pre_edit_logits = None
                        post_edit_logits = None

                        results_specificity = None
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(
                                pre_loc_logits) \
                                == len(post_loc_logits)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                # print(specificity_data[k]['probe_sentences'][
                                #                            'template_0']['label'][13:-13])
                                s_pre_perp_loss = compute_perplexity_gpt(
                                    tokenizer,
                                    pre_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                s_post_perp_loss = compute_perplexity_gpt(
                                    tokenizer,
                                    post_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})

                    pre_loc_logits = None
                    post_loc_logits = None

                elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
                    label_ids = batch["edit_inner"][0]['labels']['input_ids']
                    label_attention_mask = batch["edit_inner"][0]['labels'][
                        'attention_mask']
                    pre_perp_loss = compute_perplexity_t5(tokenizer,
                                                        pre_edit_logits,
                                                        label_ids,
                                                        label_attention_mask)
                    post_perp_loss = compute_perplexity_t5(tokenizer,
                                                        post_edit_logits,
                                                        label_ids,
                                                        label_attention_mask)

                    pre_edit_logits = None
                    post_edit_logits = None

                    results_specificity = None
                    if train_params['COMPUTE_SPECIFICITY']:
                        results_specificity = []
                        assert len(specificity_batches) == len(pre_loc_logits) \
                            == len(post_loc_logits)
                        for k in range(len(specificity_batches)):
                            s_batch = specificity_batches[k]
                            s_pre_perp_loss = compute_perplexity_t5(
                                tokenizer,
                                pre_loc_logits[k],
                                s_batch["edit_inner"][0]['labels'][
                                    'input_ids'],
                                s_batch["edit_inner"][0]['labels'][
                                    'attention_mask']
                            )

                            s_post_perp_loss = compute_perplexity_t5(
                                tokenizer,
                                post_loc_logits[k],
                                s_batch["edit_inner"][0]['labels'][
                                    'input_ids'],
                                s_batch["edit_inner"][0]['labels'][
                                    'attention_mask']
                            )

                            results_specificity.append(
                                {'pre': s_pre_perp_loss[0],
                                'post': s_post_perp_loss[0]})

                    pre_loc_logits = None
                    post_loc_logits = None
                elif train_params['BASE_MODEL'] in ['llama-7b']:
                    # print('MADE IT IN HERE!')

                    if edit_method == 'prepend_def':
                        pre_perp_loss = compute_perplexity_llama(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['probe_sentence'][
                                'input_ids'],
                            batch["edit_inner"][j]['probe_sentence'][
                                'attention_mask'],
                            batch["edit_inner"][j]['probe_sentence'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        post_perp_loss = compute_perplexity_llama(
                            tokenizer,
                            post_edit_logits,
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['input_ids'],
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['attention_mask'],
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence'],
                            batch_prepended_def["edit_inner"][j][
                                'left_context_ps'],
                            batch_prepended_def["edit_inner"][j][
                                'right_context_ps']
                        )
                        pre_edit_logits = None
                        post_edit_logits = None
                        results_specificity = None
                        # print('fourth place')
                        # print(type(specificity_batches))
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            # print(type(specificity_batches))
                            # print(type(pre_edit_dict))
                            # print(type(post_edit_dict))
                            # print(len(specificity_batches))
                            # print(len(pre_edit_dict))
                            # print(len(post_edit_dict))
                            assert len(specificity_batches) == len(
                                pre_edit_dict) \
                                == len(post_edit_dict)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                # print(specificity_data[k]['probe_sentences'][
                                #                            'template_0']['label'][13:-13])
                                s_pre_perp_loss = compute_perplexity_llama(
                                    tokenizer,
                                    pre_edit_dict[k],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['probe_sentence'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                s_post_perp_loss = compute_perplexity_llama(
                                    tokenizer,
                                    post_edit_dict[k],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['probe_sentence'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})
                        # print(pre_perp_loss)
                        # print(post_perp_loss)

                    else:
                        pre_perp_loss = compute_perplexity_llama(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['labels']['input_ids'],
                            batch["edit_inner"][j]['labels']['attention_mask'],
                            batch["edit_inner"][j]['labels'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        post_perp_loss = compute_perplexity_llama(
                            tokenizer,
                            post_edit_logits,
                            batch["edit_inner"][j]['labels']['input_ids'],
                            batch["edit_inner"][j]['labels']['attention_mask'],
                            batch["edit_inner"][j]['labels'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        pre_edit_logits = None
                        post_edit_logits = None

                        results_specificity = None
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(
                                pre_loc_logits) \
                                == len(post_loc_logits)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                # print(specificity_data[k]['probe_sentences'][
                                #                            'template_0']['label'][13:-13])
                                s_pre_perp_loss = compute_perplexity_llama(
                                    tokenizer,
                                    pre_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                s_post_perp_loss = compute_perplexity_llama(
                                    tokenizer,
                                    post_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})

                    pre_loc_logits = None
                    post_loc_logits = None


                else:
                    raise NotImplementedError

                output['pre'] = pre_perp_loss[0]
                output['post'] = post_perp_loss[0]
                # output['sim_scores'] = {
                #     'bleu_score': bleu_score,
                #     'bert_score': bert_score,
                #     'bleurt_score': bleurt_score,
                #     'meteor_score': meteor_score
                # }
                output['specificity'] = results_specificity
                all_outputs.append(output)
                # print(output)
                # print()
                # # if not pre_perp_loss[0][1]:
                #     print(ex)
        # print('one pass!')    
    return all_outputs


def group_examples(data):
    data_d = defaultdict(list)
    for ex in data:
        ent_id = '_'.join(ex['ex_id'].split('_')[:2])
        data_d[ent_id].append(ex)
    return list(data_d.items())


def run_experiment(ki_method,
                   ft_model_name,
                   dataset_name,
                   data_files,
                   device,
                   train_params,
                   random_def=None,
                   oracle_ft=False):

    outputs_d = {}
    for data_file in data_files:
        data = load_json(data_file)
        print(data_file, len(data))

        if dataset_name == 'ecbd':
            specificity_data = None
            witheld_data = None
            specificity_data = load_json(SPECIFICITY_DATA_PATH)
            witheld_data = load_json(WITHELD_DATA_PATH)
            if 'gpt2' in train_params['BASE_MODEL']:
                data = [format_gpt2_data(ex) for ex in data]
                specificity_data = [
                    format_gpt2_data(ex) for ex in specificity_data]
            elif 'gpt' in train_params['BASE_MODEL']:
                data = [format_gpt_data(ex) for ex in data]
                specificity_data = [format_gpt_data(ex) for ex in specificity_data]
                witheld_data = [format_gpt_data(ex) for ex in witheld_data]
            # For ECBD, we group examples by entities and finetune only once per
            # entity. This is unnecessary for specificity data.
            data = group_examples(data)
            all_outputs = run_edit_ecbd(
                data,
                dataset_name,
                ki_method,
                device,
                train_params,
                model_name=ft_model_name,
                random_def=random_def,
                oracle_ft=oracle_ft,
                specificity_data=specificity_data,
                witheld_data=witheld_data)

        else:  # Entity Inferences
            if 'gpt2' in train_params['BASE_MODEL']:
                data = [format_gpt2_data_entity_inferences(ex) for ex in data]
            elif 'gpt' in train_params['BASE_MODEL']:
                data = [format_gpt_data_entity_inferences(ex) for ex in data]
            all_outputs = run_edit_entity_inference(
                data,
                dataset_name,
                ki_method,
                device,
                train_params,
                model_name=ft_model_name,
                random_def=random_def)

        # Aggregate results
        outputs_d[data_file] = all_outputs

        torch.cuda.empty_cache()

    return outputs_d
