from datetime import datetime
import os
import sys
import argparse
import json
import wandb
from result_utils import compute_aggregate_metrics, process_intervention_results, process_intervention_results_gpt3, compute_aggregate_metrics_for_col
from intervention_models import load_model
from transformers import (
    GPT2Tokenizer, BertTokenizer, AutoTokenizer
)
from interventions.intervention import construct_intervention_prompts, INTERVENTION_TYPES_SINGLE_RESULT, INTERVENTION_TYPES

def run_all(args):
    print("Model:", args.model)

    print('args.intervention_types', args.intervention_types)

    # Set up folder if it does not exist
    dt_string = datetime.now().strftime("%Y%m%d")
    base_path = os.path.join(args.out_dir, "results", dt_string)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    wandb_name = ('random-' if args.random_weights else '') + \
                    args.model + '-' + \
                    args.representation + '-' + \
                    args.prompt
    wandb.init(project='causalCL', name=wandb_name, notes='', dir=base_path,
               settings=wandb.Settings(start_method='fork'), mode=args.wandb_mode)

    args_to_log = vars(args).copy()
    print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    wandb.config.update(args_to_log)
    del args_to_log

    # Initialize Model and Tokenizer
    model = load_model(args)
    tokenizer_class = (GPT2Tokenizer if model.is_gpt2 or model.is_gptneo else
                    BertTokenizer if model.is_bert else
                    AutoTokenizer if model.is_gptj else
                     None)
    if tokenizer_class is not None:
        tokenizer = tokenizer_class.from_pretrained(args.model)
    else:
        tokenizer = None

    if model.is_bert:
        model.set_st_ids(tokenizer)

    if model.is_gpt2:
        tokenizer.pad_token='<pad>'

    if not model.is_gpt3:
        model.set_vocab_subset(tokenizer, args.representation, args.max_n)

    for itype in args.intervention_types:
        if itype not in INTERVENTION_TYPES:
            raise Exception('Intervention type not defined: {}'.format(itype))

        print("\t Running with intervention: {}".format(itype))

        interventions = construct_intervention_prompts(tokenizer, itype, args)

        print('================== INTERVENTIONS ==================')
        for intervention in interventions[:4]:
            print('v.base_string', intervention.base_string)
            print('v.alt_string', intervention.alt_string)

        multitoken = (args.representation == 'words' and args.max_n > 20)
        if multitoken:
            print('Multitoken setting')

        # Run actual exp
        intervention_results = model.intervention_experiment(interventions, multitoken=multitoken)

        single_result = itype in INTERVENTION_TYPES_SINGLE_RESULT
        if model.is_gpt3:
            df = process_intervention_results_gpt3(interventions, intervention_results, args.representation, single_result=single_result)
            metrics_dict = {}
            metric_dict = compute_aggregate_metrics_for_col(df['confidence_change'])
            metrics_dict['confidence_change'] = metric_dict

        else:
            df = process_intervention_results(interventions, intervention_results, args.max_n, args.representation, single_result=single_result)
            metrics_dict = compute_aggregate_metrics(df, single_result=single_result)

        print(json.dumps(metrics_dict, indent=4))
        metrics_dict = { 'int{}_'.format(itype) + k : v for k, v in metrics_dict.items()}
        wandb.run.summary.update(metrics_dict)


        random = ['random'] if args.random_weights else []
        fcomponents = random + [args.model, itype]
        fname = "_".join(fcomponents).replace('/','-')

        out_path = os.path.join(base_path, fname+".csv")
        print('out_path: ', out_path)
        df.to_csv(out_path)


if __name__ == "__main__":
    if not (len(sys.argv) == 16):
        print("USAGE: python ", sys.argv[0], 
                "<model> <device> <out_dir> <random_weights> <representation> <seed> <prompt> <path_to_num_data> <examples> <wandb_mode> <intervention_type> <examples_per_template> <transformers_cache_dir> <path_to_dict>")

    PARAMETERS = {
        'wandb_mode': sys.argv[10],

        'model' : sys.argv[1],  # distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
        'device' : sys.argv[2],  # cpu vs cuda
        'out_dir' : sys.argv[3],  # dir to write results
        'random_weights' : sys.argv[4] == 'random',  # true or false
        'representation' : sys.argv[5],  # arabic, words
        'seed' : int(sys.argv[6]),  # to allow consistent sampling
        'prompt' : sys.argv[7],  # question vs statement. Whether the prompt should be formulated as a question or statement
        'path_to_num_data' : sys.argv[8],  # path to the mwps csv file
        'max_n' : int(sys.argv[9]),  # number of examples to try, 0 for all
        'examples_per_template' : int(sys.argv[12]),

        'transformers_cache_dir': sys.argv[13],

        'path_to_dict' : sys.argv[14],

        'intervention_types' : sys.argv[11].split('-'), # list
    }

    print('Arguments:', PARAMETERS)

    args = argparse.Namespace(**PARAMETERS)
        
    run_all(args)
