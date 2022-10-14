from intervention_models.intervention_model import Model
from intervention_models.api_intervention_model import ApiModel

def load_model(args):
    if args.model.startswith('gpt3'):
        return ApiModel(args.model)
    else:
        return Model(device=args.device, model_version=args.model, random_weights=args.random_weights,
                  transformers_cache_dir=args.transformers_cache_dir)