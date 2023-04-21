import torch
from torch.nn import CrossEntropyLoss
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import copy 
from .data_utils import CATEGORY_MAP, CustomDataSetClass, random_mask
from .data_utils import particle_mask, nll_mask
import sys
sys.path.insert(1, '/data/shankar/ping_knowledge_injection/src')
from distill_gpt import gpt_distill, gpt_distill_generate, gpt_distill_next_token, gpt_distill_after_entity_span

def train(tokenizer, model, device, loader, optimizer):
    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    losses = []
    for _, data in enumerate(loader, 0):

        if 'gpt' in model.name_or_path:
            bsize = data['target_ids'].size(0)
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['input_mask'].to(device, dtype=torch.long)
            y = data['target_ids'].to(device, dtype=torch.long)
            lm_labels = y.clone().detach()
            lm_labels[y == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=ids,
                            attention_mask=mask,
                            labels=lm_labels)
        else:
            bsize = data['target_ids'].size(0)
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['input_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids,
                            attention_mask=mask,
                            decoder_input_ids=y_ids,
                            labels=lm_labels)
        loss = outputs[0]

        lm_logits = outputs[1]

        input_str = [tokenizer.decode(i, skip_special_tokens=True) for i in ids]

        target_len = (data['target_ids'] != 0).sum(-1)

        label_str = [tokenizer.convert_ids_to_tokens(data['target_ids'][i])
                     [:target_len[i]] for i in range(bsize)]

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        per_token_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(bsize, -1)

        per_token_loss = \
            [per_token_loss[i, :target_len[i] - 1].cpu().detach().numpy() for i
            in range(bsize)]

        losses.append((loss.item(), per_token_loss, input_str, label_str))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


def finetuning(model, tokenizer, ex, model_params, device):
    if 'gpt' in model.name_or_path:
        data_d = {'input_text': [ex['definition']],
                  'target_text': [ex['definition']]}

        if model_params['AUGMENT']:
            n_augmented_sentences = len(
                ex['additional_sentences'][:model_params['NUM_AUGMENT']])
            for i in range(n_augmented_sentences):
                masked_sentence, target = ex['additional_sentences'][i]
                _sentence = masked_sentence.replace('<extra_id_0>',
                                                    target[13:-13])
                data_d['input_text'].append(_sentence)
                data_d['target_text'].append(_sentence)

        # if model_params['CONCEPTNET']:
        #     n_augmented_sentences = len(ex['conceptnet'])
        #     for i in range(n_augmented_sentences):
        #         for _ in range(model_params['NUM_AUGMENT']):
        #             _sentence = ex['conceptnet'][i]
        #             _sentence = _sentence.replace('<ENT_NAME>', ex['ent_str'])
        #             data_d['input_text'].append(_sentence)
        #             data_d['target_text'].append(_sentence)

        if 'AUGMENT_GENERIC' in model_params and model_params[
            'AUGMENT_GENERIC']:
            n_augmented_sentences = len(ex['generic_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    _sentence = ex['generic_sentences'][i].replace(
                        '<ENT_NAME>', ex["def_target"].lstrip(
                            '<extra_id_0> ').rstrip(' <extra_id_1>'))
                    data_d['input_text'].append(_sentence)
                    data_d['target_text'].append(_sentence)

        if 'AUGMENT_SPECIFIC' in model_params and model_params[
            'AUGMENT_SPECIFIC']:
            n_augmented_sentences = len(ex['specific_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    _sentence = ex['specific_sentences'][i].replace(
                        '<ENT_NAME>',ex["def_target"].lstrip(
                            '<extra_id_0> ').rstrip(' <extra_id_1>'))
                    data_d['input_text'].append(_sentence)
                    data_d['target_text'].append(_sentence)

        if model_params['X_IS_Y']:
            _sentence = '{} is a {}.'.format(ex['ent_str'],
                                             CATEGORY_MAP[ex['category']])
            data_d['input_text'].append(_sentence)
            data_d['target_text'].append(_sentence)

        if model_params['MEMORY_RETRIEVAL']:

            _sentences = retrieve_memory(
                model, tokenizer, ex['definition'], ex['ent_str'], device,
                seed=2022)
            data_d['input_text'].extend(_sentences)
            data_d['target_text'].extend(_sentences)

        if model_params['TRAIN_ON_PROBE']:
            _probe_sentence = ex['probe_sentences']['template_0'][
                                 'probe_sentence'].strip(' <|endoftext|>')
            _label = ex['probe_sentences']['template_0']['label']
            _probe_sentence = _probe_sentence.replace(
                '<extra_id_0>', _label[13:-13])

            data_d = {'input_text': [_probe_sentence],
                      'target_text': [_probe_sentence]}

    else:
        if model_params['USE_NLL_MASKS']:
            data_d = {'input_text': [], 'target_text': []}
            definition_sentence = ex['definition'].replace('<extra_id_0>',
                                                           ex['def_target'][
                                                           13:-13])
            masked_sentences, targets = nll_mask(tokenizer, definition_sentence,
                                                 ex['ent_str'], model, device,
                                                 topk=model_params['TOPK'])
            for masked_sentence, target in zip(masked_sentences, targets):
                data_d['input_text'].append(masked_sentence)
                data_d['target_text'].append(target)
        else:
            data_d = {'input_text': [ex['definition']],
                      'target_text': [ex['def_target']]}

        mask_func = random_mask if model_params[
                                       'MASKING_STRATEGY'] == 'random' else particle_mask

        if model_params['AUGMENT']:
            n_augmented_sentences = len(
                ex['additional_sentences'][:model_params['NUM_AUGMENT']])
            for i in range(n_augmented_sentences):
                masked_sentence, target = ex['additional_sentences'][i]
                data_d['input_text'].append(masked_sentence)
                data_d['target_text'].append(target)

        # if model_params['CONCEPTNET']:
        #     n_augmented_sentences = len(ex['conceptnet'])
        #     for i in range(n_augmented_sentences):
        #         for _ in range(model_params['NUM_AUGMENT']):
        #             masked_sentence, target = mask_func(ex['conceptnet'][i])
        #             data_d['input_text'].append(
        #                 masked_sentence.replace('<ENT_NAME>',
        #                                         ex["def_target"].lstrip(
        #                                             '<extra_id_0> ').rstrip(
        #                                             ' <extra_id_1>')))
        #             data_d['target_text'].append(target)

        if 'AUGMENT_GENERIC' in model_params and model_params[
            'AUGMENT_GENERIC']:
            n_augmented_sentences = len(ex['generic_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    masked_sentence, target = mask_func(ex['generic_sentences'][i])
                    data_d['input_text'].append(masked_sentence.replace('<ENT_NAME>', ex["def_target"].lstrip('<extra_id_0> ').rstrip(' <extra_id_1>')))
                    data_d['target_text'].append(target)

        if 'AUGMENT_SPECIFIC' in model_params and model_params[
            'AUGMENT_SPECIFIC']:
            n_augmented_sentences = len(ex['specific_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    masked_sentence, target = mask_func(ex['specific_sentences'][i])
                    data_d['input_text'].append(masked_sentence.replace('<ENT_NAME>', ex["def_target"].lstrip('<extra_id_0> ').rstrip(' <extra_id_1>')))
                    data_d['target_text'].append(target)

        if model_params['X_IS_Y']:
            masked_sentence = '{} is a <extra_id_0>.'.format(ex['ent_str'])
            target = '<extra_id_0> {} <extra_id_1>'.format(
                CATEGORY_MAP[ex['category']])
            data_d['input_text'].append(masked_sentence)
            data_d['target_text'].append(target)

    training_set = CustomDataSetClass(
        data_d,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"]
    )

    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }

    set_seed(model_params["SEED"])

    training_loader = DataLoader(training_set, **train_params)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=model_params["LEARNING_RATE"])

    all_losses = []
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        losses = train(tokenizer, model, device, training_loader, optimizer)
        all_losses.append(losses[0][0])

    model.eval()
    return model, all_losses


def retrieve_memory(model, tokenizer, definition, ent_str, device, seed=2022):

    questions = [
        '\n\nQuestion: What is similar to {}?\n\nAnswer:',
        '\n\nQuestion: When did {} happen?\n\nAnswer:',
        '\n\nQuestion: Who involved in {}?\n\nAnswer:',
        '\n\nQuestion: Where is the origin of {}?\n\nAnswer:',
    ]

    def separate_sentence(s):
        ans = s.split('\n\n')[2]
        assert ans.startswith('Answer:')
        ans = ans.lstrip('Answer: ')
        return ans

    sents = []
    for i, question in enumerate(questions):
        prompt = definition + question.format(ent_str)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.manual_seed(seed)
        gen_tokens = model.generate(
            input_ids,
            num_beams=1,
            num_beam_groups=1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.9,
            max_length=128,
            pad_token_id=tokenizer.eos_token_id)

        gen_text = tokenizer.batch_decode(gen_tokens)
        ans = separate_sentence(gen_text[0])
        if ans:
            sents.append(ans)

    return sents
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
def apply_ft_distill_gpt(model_raw, teacher_type, teacher, tokenizer, contexts, unmasked_probe_set,  device, lr, num_steps, max_length, sample_temperature, softmax_temperature,  top_p, 
                   num_probes, repetition_penalty, top_k, length_penalty, beam_search, num_updates, batch_size, ent_strs, after_ent_span=None,specificity_batches=None, dataset_name=None):

    model_ft = copy.deepcopy(model_raw)
    for i in range(len(unmasked_probe_set)):
        counter=0
        for probey in unmasked_probe_set[i]: #distill information from each probe to model
            model_raw = gpt_distill_after_entity_span(model=model_ft, teacher_type=teacher_type, teacher=teacher, tokenizer=tokenizer, context=contexts[i],ent_str=ent_strs[i], 
                                                    probey=probey, device=device, dataset_name=dataset_name, lr=lr, num_steps=num_steps, max_length=max_length, 
                                                    top_p=top_p, repetition_penalty=repetition_penalty, softmax_temperature=softmax_temperature, 
                                                    sample_temperature=sample_temperature, top_k=top_k, length_penalty=length_penalty, beam_search=beam_search)
            if model_raw!=False:
                model_ft = model_raw
                counter+=1
            if counter==num_updates:
                break

    model_ft.eval()

    return model_ft