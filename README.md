# Causal Analysis of Mathematical Reasoning in Neural Language Models

This repository contains the code for the paper "A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models".

_Alessandro Stolfo*, Zhijing Jin*, Kumar Shridhar, Bernhard Schölkopf, Mrinmaya Sachan_


## Requirements
- `python==3.9`
- `pytorch==1.11`
- `pandas==1.4`
- `spacy==3.3.1`
- `transformers==4.20.0`
- `wandb`

## Experiments
The intervention experiments can be run by setting the desired parameters in `run_numeracy_exp.sh` and then executing `sh run_numeracy_exp.sh`.

### Intervention Types
The parameters representing the different intervention/effect combinations are the following:
- `1b`: DCE(N --> R)
- `2`: TCE(N --> R)
- `3`: DCE(S --> R)
- `4`: TCE(T --> R)

### Models
The models on which this repo was tested are:
- `distilgpt2`
- `gpt2`
- `gpt2-medium`
- `gpt2-large`
- `gpt2-xl`
- `EleutherAI/gpt-neo-1.3B`
- `EleutherAI/gpt-neo-2.7B`
- `EleutherAI/gpt-j-6B`
- `gpt3/text-davinci-002`

For GPT-3, the organization key and secret key are necessary to access the [OpenAI APIs](https://openai.com/api/).
If you want to experiment with it, paste the keys as the first (organization) and second (secret) lines in the `openai_keys.txt` file.

All the other models are accessed through [Huggingface Transformers](https://huggingface.co).

GPT-Neo-2.7B requires a GPU with at least 24GB of memory, and GPT-J-6B requires a GPU with at least 32 GB of memory. 

### Heatmaps
To obtain the data needed to plot the heatmaps reported in the paper, execute
```
python heatmap_experiments/heatmap_experiment.py [model] [device] [out_dir] not_random arabic [seed] statement [data_path] [max_n] disabled + [n_templates]"
```
This will store the average probabilities assigned to each possible groud-truth result in the range 0,...,`max_n` as a `.csv` file.


## Other Papers / Resources

Part of this repo is based on the code for [Finlayson et al. (2020)](https://github.com/mattf1n/lm-intervention)

