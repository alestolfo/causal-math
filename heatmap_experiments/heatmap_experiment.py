from datetime import datetime
import random
import os
import sys
import argparse
import json
import numpy as np
from intervention_models.intervention_model import Model
from transformers import (
    GPT2Tokenizer, BertTokenizer, AutoTokenizer
)
from numerical_utils import construct_numerical_templates
from number_generator import NumberGenerator
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import torch


def run_all(args, shuffle_funct):
    print("Model:", args.model)

    # Set up folder if it does not exist
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string + "_neuron_intervention"
    base_path = os.path.join(args.out_dir, "results", folder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    args_to_log = vars(args).copy()
    print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    del args_to_log

    # Initialize Model and Tokenizer
    model = Model(device=args.device, model_version=args.model, random_weights=args.random_weights,
                  transformers_cache_dir=args.transformers_cache_dir)
    tokenizer_class = (GPT2Tokenizer if model.is_gpt2 or model.is_gptneo else
                       BertTokenizer if model.is_bert else
                       AutoTokenizer if model.is_gptj else
                       None)
    tokenizer = tokenizer_class.from_pretrained(args.model)

    if model.is_bert:
        model.set_st_ids(tokenizer)

    if model.is_gpt2:
        tokenizer.pad_token = '<pad>'

    model.set_vocab_subset(tokenizer, args.representation, args.max_n)

    print("\t Running with operation: {}".format(args.operation))

    templates = construct_numerical_templates(path_to_data=args.path_to_num_data,
                                              path_to_dict=args.path_to_dict,
                                              statements=('statement' in args.prompt),
                                              questions=('question' in args.prompt))
    shuffle_funct(templates)
    entries = {}
    n_template_used = 0
    for t_dict in templates:
        eq = t_dict['eq']
        if t_dict['n_vars'] != 2 or eq[0] != args.operation:
            continue

        print('t_dict', t_dict)

        op, placeholder0, placeholder1 = eq.split(' ')

        print('equation', eq)

        assert args.operation == op

        second_part = 'statement' if 'statement' in args.prompt else 'question'
        temp = (t_dict['body'] + ' ' + t_dict[second_part])

        number_generator = NumberGenerator(args.max_n, args.seed, args.min_n, force_generate=True)

        op_triples_list = number_generator.get_triples_dict()[op]

        for i, triple in enumerate(tqdm(op_triples_list, desc='triples')):
            x, y, res = triple
            x_str = str(x)
            y_str = str(y)
            prompt = temp.replace(placeholder0, x_str).replace(placeholder1, y_str)
            tok_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            tok_prompt = torch.LongTensor(tok_prompt)

            logits_subset, probs_subset = model.get_distribution_for_examples(tok_prompt.unsqueeze(0))
            normalized_probs = F.softmax(torch.tensor(logits_subset), dim=-1)

            p_res = normalized_probs[res].item()

            if (x,y) not in entries:
                entries[(x, y, res)] = [p_res]
            else:
                entries[(x, y, res)].append(res)

        n_template_used += 1
        if n_template_used >= args.n_templates:
            break

    df_rows = []
    for triple, p_res_list in entries.items():
        x, y, res = triple
        avg_prob = np.mean(p_res_list)
        df_row = {'n1': x, 'n2': y, 'res': res, 'p_res' : avg_prob}
        df_rows.append(df_row)

    df = pd.DataFrame(df_rows)

    # Generate file name
    random = ['random'] if args.random_weights else []
    fcomponents =  ['heatmap', str(args.representation), args.operation, str(args.min_n), str(args.max_n), args.model, str(args.n_templates)] + random
    fname = "_".join(fcomponents).replace('/', '-')
    # Finally, save each exp separately
    out_path = os.path.join(base_path, fname + ".csv")
    print('out_path: ', out_path)
    df.to_csv(out_path)


if __name__ == "__main__":

    PARAMETERS = {
        'wandb_mode': sys.argv[10],

        'model': sys.argv[1],  # distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
        'device': sys.argv[2],  # cpu vs cuda
        'out_dir': sys.argv[3],  # dir to write results
        'random_weights': sys.argv[4] == 'random',  # true or false
        'representation': sys.argv[5],  # arabic, words
        'seed': int(sys.argv[6]),  # to allow consistent sampling
        'prompt': sys.argv[7],
        # question vs statement. Whether the prompt should be formulated as a question or statement
        'path_to_num_data': sys.argv[8],  # path to the mwps csv file
        'max_n': int(sys.argv[9]),  # number of examples to try, 0 for all
        'min_n': 0,
        'n_templates': int(sys.argv[12]),

        'transformers_cache_dir': sys.argv[13],

        'path_to_dict': sys.argv[14],

        'operation': sys.argv[11],  #  + - * /
    }

    print('Arguments:', PARAMETERS)

    args = argparse.Namespace(**PARAMETERS)

    random.seed(args.seed)
    shuffle_funct = random.shuffle


    run_all(args, shuffle_funct)
