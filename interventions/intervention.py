from numerical_utils import construct_numerical_templates
import torch
from interventions.number_interventions import *
from interventions.template_interventions import *

INTERVENTION_TYPES_TWO_RESULTS = ['0', '1', '1b', '1c', '4']
INTERVENTION_TYPES_SINGLE_RESULT = ['2', '3', '10', '11']
INTERVENTION_TYPES = INTERVENTION_TYPES_TWO_RESULTS + INTERVENTION_TYPES_SINGLE_RESULT


def construct_intervention_prompts(tokenizer, intervention_type, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    templates = construct_numerical_templates(path_to_data=args.path_to_num_data,
                                              path_to_dict=args.path_to_dict,
                                              statements=('statement' in args.prompt),
                                              questions=('question' in args.prompt))

    if intervention_type == '1':
        # change one number -> change result
        interventions = change_numbers_interventions(templates, tokenizer, args)
    elif intervention_type == '1b':
        # change two numbers -> change result
        interventions = change_numbers_interventions(templates, tokenizer, args, n_changed_vars='two')
    elif intervention_type == '1c':
        # change all numbers -> change result
        interventions = change_numbers_interventions(templates, tokenizer, args, n_changed_vars='all')
    elif intervention_type == '2':
        # change all numbers -> same result
        interventions = change_numbers_interventions(templates, tokenizer, args, change_result=False)
    elif intervention_type == '3':
        # change template -> same result
        interventions = change_template_same_result_interventions(templates, tokenizer, args)
    elif intervention_type == '4':
        # change template -> change result
        interventions = change_template_different_result_interventions(templates, tokenizer, args)
    elif intervention_type == '0':
        # add random words instead of numbers -> change result
        interventions = change_numbers_interventions(templates, tokenizer, args, control=True)
    elif intervention_type == '10':
        # add useless sentence
        interventions = add_sentence_same_result_interventions(templates, tokenizer, args)
    elif intervention_type == '11':
        # add useless sentence with a number
        interventions = add_sentence_same_result_interventions(templates, tokenizer, args, add_number=True)
    else:
        raise Exception('intervention_type not defined {}'.format(intervention_type))

    return interventions