import torch
from transformers import GPT2Tokenizer, BertTokenizer


class Intervention():
    '''
    Wrapper for all the possible interventions
    '''
    def __init__(self,
                 tokenizer,
                 template_id,
                 base_temp: str,
                 alt_temp: str,
                 equation: str,
                 vars_base,
                 vars_alt,
                 n_vars,
                 multitoken=False,
                 device='cpu'):
        super()
        self.device = device
        self.multitoken = multitoken

        if isinstance(tokenizer, BertTokenizer):
            base_temp = base_temp + ' [MASK]'
            alt_temp = alt_temp + ' [MASK]'

        self.template_id = template_id
        self.n_vars = n_vars

        # All the initial strings
        self.base_string = base_temp
        self.alt_string = alt_temp

        self.equation = equation
        self.vars_base = vars_base
        self.vars_alt = vars_alt

        self.enc = tokenizer

        if self.enc is not None:
            self.base_string_tok = self.enc.encode(base_temp, add_special_tokens=False)
            self.alt_string_tok = self.enc.encode(alt_temp, add_special_tokens=False)

            self.base_string_tok = torch.LongTensor(self.base_string_tok).to(device)
            self.alt_string_tok = torch.LongTensor(self.alt_string_tok).to(device)


    def set_results(self, res_base, res_alt):
        self.res_base_string  = res_base
        self.res_alt_string = res_alt

        if self.enc is not None:
            # 'a ' added to input so that tokenizer understands that first word
            # follows a space.
            self.res_base_tok = self.enc.tokenize('a ' + res_base)[1:]
            self.res_alt_tok = self.enc.tokenize('a ' + res_alt)[1:]
            if not self.multitoken:
                assert len(self.res_base_tok) == 1, '{} - {}'.format(self.res_base_tok, res_base)
                assert len(self.res_alt_tok) == 1, '{} - {}'.format(self.res_alt_tok, res_alt)

            self.res_base_tok = self.enc.convert_tokens_to_ids(self.res_base_tok)
            self.res_alt_tok = self.enc.convert_tokens_to_ids(self.res_alt_tok)

    def set_result(self, res):
        self.res_string = res

        if self.enc is not None:
            self.res_tok = self.enc.tokenize('a ' + res)[1:]
            if not self.multitoken:
                assert (len(self.res_tok) == 1)