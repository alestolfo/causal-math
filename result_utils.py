import pandas as pd
import numpy as np
from numerical_utils import convert_to_words

def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx:min(ndx + bsize, total_len)])

def kl_div(a,b):
    return sum([a[i] * np.log(a[i] / b[i]) if a[i] > 0 else 0 for i in range(len(a))])

def tv_distance(a,b):
    return np.sum(np.abs(a - b)) / 2

def d_inf(a,b):
    return np.max(np.log(np.maximum(a/b,b/a)))

def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def compute_relative_confidence_change(distrib_base, distrib_alt, c1, c2):
    candidate1_base_prob = distrib_base[c1]
    candidate2_base_prob = distrib_base[c2]
    candidate1_alt1_prob = distrib_alt[c1]
    candidate2_alt1_prob = distrib_alt[c2]
    base_error = candidate2_base_prob / candidate1_base_prob
    alt_error = candidate1_alt1_prob / candidate2_alt1_prob
    total_effect = 1 / (alt_error * base_error) - 1

    return  total_effect

def process_intervention_results_gpt3(interventions, intervention_results, representation, single_result=True):
    results = []
    for example in intervention_results:
        base_logprobs, alt_logprobs = intervention_results[example]

        base_logprobs = [dict(lp) for lp in base_logprobs]
        alt_logprobs = [dict(lp) for lp in alt_logprobs]

        intervention = interventions[example]

        metric_dict = {
            'example': example,
            'template_id': intervention.template_id,
            'n_vars': intervention.n_vars,
            'base_string': intervention.base_string,
            'alt_string1': intervention.alt_string,
            'operation': intervention.equation[0],
            'equation': intervention.equation,
            'vars_base': intervention.vars_base,
            'vars_alt': intervention.vars_alt,
            'base_tok_prob_dict': base_logprobs,
            'alt_tok_prob_dict': alt_logprobs,
            'base_next_tok_pred' : base_logprobs[0],
            'alt_next_tok_pred': alt_logprobs[0],
        } # TODO find a better way to store OpenaiObject

        if single_result:
            metric_dict['res'] = intervention.res_string

            if representation == 'arabic':
                res = intervention.res_string
            else:
                raise Exception('Representation unknown: {}'.format(representation))

            prob_res_base = None
            base_top5_probs = []
            for tok, logprob in base_logprobs[0].items():
                prob = np.exp(logprob)
                base_top5_probs.append(prob)
                if tok.strip() == res:
                    prob_res_base = prob

            prob_res_alt = None
            alt_top5_probs = []
            for tok, logprob in alt_logprobs[0].items():
                prob = np.exp(logprob)
                alt_top5_probs.append(prob)
                if tok.strip() == res:
                    prob_res_alt = np.exp(logprob)

            # Confidence change
            if not prob_res_alt or not prob_res_base:
                confidence_change = np.NaN
            else:
                base_confidence_change = (prob_res_base - prob_res_alt) / prob_res_alt
                alt_confidence_change = (prob_res_alt - prob_res_base) / prob_res_base
                confidence_change = max(base_confidence_change, alt_confidence_change)

        else:
            metric_dict['res_base'] = intervention.res_base_string
            metric_dict['res_alt'] = intervention.res_alt_string

            if representation == 'arabic':
                res_base = intervention.res_base_string
                res_alt = intervention.res_alt_string
            else:
                raise Exception('Representation unknown: {}'.format(representation))

            prob_base_res_base = None
            prob_base_res_alt = None
            base_topk_probs = []
            for tok, logprob in base_logprobs[0].items():
                prob = np.exp(logprob)
                base_topk_probs.append(prob)
                if tok.strip() == res_base:
                    prob_base_res_base = prob
                if tok.strip() == res_alt:
                    prob_base_res_alt = prob

            prob_alt_res_alt = None
            prob_alt_res_base = None
            alt_topk_probs = []
            for tok, logprob in alt_logprobs[0].items():
                prob = np.exp(logprob)
                alt_topk_probs.append(prob)
                if tok.strip() == res_alt:
                    prob_alt_res_alt = prob
                if tok.strip() == res_base:
                    prob_alt_res_base = prob

            base_min_top_prob = min(base_topk_probs)
            alt_min_top_prob = min(alt_topk_probs)

            # Confidence change
            if prob_base_res_base:
                if not prob_alt_res_base:
                    prob_alt_res_base = alt_min_top_prob
                base_confidence_change = (prob_base_res_base - prob_alt_res_base) / prob_alt_res_base
            else:
                base_confidence_change = 0


            if prob_alt_res_alt:
                if not prob_base_res_alt:
                    prob_base_res_alt = base_min_top_prob
                alt_confidence_change = (prob_alt_res_alt - prob_base_res_alt) / prob_base_res_alt
            else:
                alt_confidence_change = 0

            confidence_change = (base_confidence_change + alt_confidence_change) / 2

        metric_dict['confidence_change'] = confidence_change

        results.append(metric_dict)

    return pd.DataFrame(results)



def process_intervention_results(interventions, intervention_results, max_n, representation, single_result=True):
    results = []
    for example in intervention_results:
        logits_base, logits_alt, _, _ = intervention_results[example]

        intervention = interventions[example]

        normalized_base = softmax(np.array(logits_base))
        normalized_alt = softmax(np.array(logits_alt))

        js_div = (kl_div(normalized_base, (normalized_alt + normalized_base) / 2) + kl_div(normalized_alt, (normalized_alt + normalized_base) / 2)) / 2
        tv_norm = tv_distance(normalized_base, normalized_alt)
        l_inf_div = d_inf(normalized_base, normalized_alt)

        # Causal effect
        pred_base = normalized_base.argmax()
        pred_alt = normalized_alt.argmax()
        causal_effect = np.abs(pred_base - pred_alt)

        metric_dict = {
            'example': example,
            'template_id': intervention.template_id,
            'n_vars' : intervention.n_vars,
            'base_string': intervention.base_string,
            'alt_string1': intervention.alt_string,
            'operation': intervention.equation[0],
            'equation': intervention.equation,
            'vars_base': intervention.vars_base,
            'vars_alt': intervention.vars_alt,
            'pred_base': pred_base,
            'pred_alt': pred_alt,

            'distrib_base': normalized_base,
            'distrib_alt': normalized_alt,
            'js_div': js_div,
            'tv_norm': tv_norm,
            'l_inf_div': l_inf_div,
            'causal_effect': causal_effect
        }

        if single_result:
            metric_dict['res'] = intervention.res_string
            if representation == 'arabic':
                res = int(intervention.res_string)
            elif representation == 'words':
                words_to_n = {convert_to_words(str(i)): i for i in range(max_n + 1)}
                res = int(words_to_n[intervention.res_string])
            else:
                raise Exception('Representation unknown: {}'.format(representation))

            # Prob change
            prob_res_base = normalized_base[res]
            prob_res_alt = normalized_alt[res]
            abs_prob_change = np.abs(prob_res_alt - prob_res_base) / prob_res_base
            metric_dict['old_confidence_change'] = abs_prob_change

            # Confidence change
            base_confidence_change = (prob_res_base - prob_res_alt) / prob_res_alt
            alt_confidence_change =  (prob_res_alt - prob_res_base) / prob_res_base
            confidence_change = max(base_confidence_change, alt_confidence_change)

            abs_confidence_change = abs_prob_change

            # Error change
            error_base = np.abs(res - pred_base)
            error_alt = np.abs(res - pred_alt)
            error_change = np.abs(error_base - error_alt)

        else:
            metric_dict['res_base'] = intervention.res_base_string
            metric_dict['res_alt'] = intervention.res_alt_string
            if representation == 'arabic':
                res_base = int(intervention.res_base_string)
                res_alt = int(intervention.res_alt_string)
            elif representation == 'words':
                words_to_n = {convert_to_words(str(i)): i for i in range(max_n + 1)}
                res_base = int(words_to_n[intervention.res_base_string])
                res_alt = int(words_to_n[intervention.res_alt_string])
            else:
                raise Exception('Representation unknown: {}'.format(representation))

            prob_base_res_base = normalized_base[res_base]
            prob_base_res_alt = normalized_base[res_alt]
            prob_alt_res_base = normalized_alt[res_base]
            prob_alt_res_alt = normalized_alt[res_alt]

            # Confidence change
            base_confidence_change = (prob_base_res_base - prob_alt_res_base) / prob_alt_res_base
            alt_confidence_change = (prob_alt_res_alt - prob_base_res_alt) / prob_base_res_alt
            confidence_change = (base_confidence_change + alt_confidence_change) / 2

            # Absolute value confidence change
            abs_base_confidence_change = np.abs(prob_base_res_base - prob_alt_res_base) / prob_alt_res_base
            abs_alt_confidence_change = np.abs(prob_alt_res_alt - prob_base_res_alt) / prob_base_res_alt
            abs_confidence_change = (abs_base_confidence_change + abs_alt_confidence_change) / 2

            # relative confidence change
            relative_confidence_change = compute_relative_confidence_change(normalized_base, normalized_alt, res_base, res_alt)
            metric_dict['relative_confidence_change'] = relative_confidence_change

            # error change
            error_base = np.abs(pred_base - res_base)
            error_alt = np.abs(pred_alt - res_alt)
            error_change = np.abs(error_base - error_alt)

        metric_dict['error_change'] = error_change
        metric_dict['confidence_change'] = confidence_change
        metric_dict['abs_confidence_change'] = abs_confidence_change

        results.append(metric_dict)

    return pd.DataFrame(results)


def compute_aggregate_metrics_for_col(col):
    return {'mean' : col.mean(), 'sem' : col.sem(), 'std' : col.std()}

def compute_aggregate_metrics(df, single_result):
    metrics_dict = {}
    measures = ['causal_effect', 'js_div', 'tv_norm', 'error_change', 'confidence_change', 'abs_confidence_change']
    if not single_result:
        measures = measures + ['relative_confidence_change']

    for measure in measures:
        metric_dict = compute_aggregate_metrics_for_col(df[measure])
        metrics_dict[measure] = metric_dict

    return metrics_dict

