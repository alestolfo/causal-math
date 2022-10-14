import random
from os.path import exists
import json
from numerical_utils import factors
from copy import deepcopy

class NumberGenerator():
    def __init__(self, max_n, seed, min_n=1, for_heatmap=False):
        self.max_n = max_n
        self.min_n = min_n
        self.for_heatmap = for_heatmap
        random.seed(seed)

        self.generate_triples()

        self.immutable_triples_dict = {}

        for op, l in self.triples_dict.items():
            random.shuffle(l)
            self.immutable_triples_dict[op] = l.copy()

        # dictionary to make generation of intervention type 4 more efficient
        pairs_dict = {}
        for op, l in self.triples_dict.items():
            for triple in l:
                x, y, res = triple
                pair = (x, y)
                if pair not in pairs_dict:
                    pairs_dict[pair] = {'+' : None, '*' : None, '-' : None, '/' : None}
                pairs_dict[pair][op] = res
        # avoid using (2, 2) because changing op will lead to same result
        pairs_dict[(2, 2)] = {'+' : None, '*' : None, '-' : None, '/' : None}
        self.pairs_dict = pairs_dict


    def get_triples_dict(self):
        return self.triples_dict


    def reset_triple_lists(self):
        #self.triples_dict = deepcopy(self.immutable_triples_dict)
        for op, l in self.immutable_triples_dict.items():
            self.triples_dict[op] = l.copy()


    def generate_triples(self):
        addition_triples = []
        subtraction_triples = []
        upper_bound = self.max_n + 1 if self.for_heatmap else self.max_n
        for n1 in range(self.min_n, upper_bound):
            for n2 in range(self.min_n, self.max_n - n1 + 1):
                addition_triple = (n1, n2, n1 + n2)
                subtraction_triple = (n1 + n2, n2, n1)
                addition_triples.append(addition_triple)
                subtraction_triples.append(subtraction_triple)

        multiplication_triples = []
        division_triples = []
        for res in range(2, self.max_n + 1):
            fact_list = factors(res)
            fact_list.remove(1)
            fact_list.remove(res)
            for n1 in fact_list:
                for n2 in fact_list:
                    if n1 * n2 == res:
                        mult_triple = (n1, n2, res)
                        div_triple = (res, n2, n1)
                        multiplication_triples.append(mult_triple)
                        division_triples.append(div_triple)

        triples_dict = {
            '+' : addition_triples,
            '-' : subtraction_triples,
            '*' : multiplication_triples,
            '/' : division_triples
        }

        self.triples_dict = triples_dict


    def check_correctness(self, eq1, eq2, triple1, triple2, max_n):
        x1, y1, res1 = triple1
        x2, y2, res2 = triple2
        assert x1 in range(max_n + 1)
        assert x2 in range(max_n + 1)
        assert y1 in range(max_n + 1)
        assert y2 in range(max_n + 1)
        assert res1 in range(max_n + 1)
        assert res2 in range(max_n + 1)

        operation1_symbol = eq1[0]
        operation2_symbol = eq2[0]

        ops = []
        for operation_symbol in [operation1_symbol, operation2_symbol]:
            if operation_symbol == '+':
                op = lambda a, b: a + b
            elif operation_symbol == '-':
                op = lambda a, b: a - b
            elif operation_symbol == '*':
                op = lambda a, b: a * b
            elif operation_symbol == '/':
                op = lambda a, b: a // b
            else:
                raise Exception('Operation not defined: {}'.format(operation_symbol))
            ops.append(op)

        assert ops[0](x1, y1) == res1, f'Error while generating numbers: {x1} {operation1_symbol} {y1} != {res1}'
        assert ops[1](x2, y2) == res2, f'Error while generating numbers: {x2} {operation2_symbol} {y2} != {res2}'


    def get_two_triples_with_condition(self, condition, triples):
        if condition == 'same_result':
            cond = lambda t1, t2: t1[2] == t2[2]
        elif condition == 'one_number_fixed':
            cond = lambda t1, t2 : t1[0] == t2[0] or t1[1] == t2[1] and t1[2] != t2[2]
        elif condition == 'both_numbers_change':
            cond = lambda t1, t2: t1[0] != t2[0] and t1[1] != t2[1] and t1[2] != t2[2]
        else:
            raise Exception(f'Condition {condition} not recognized')

        triple1 = None
        triple2 = None
        for idx_1, candidate_t1 in enumerate(triples):
            for idx_2, candidate_t2 in enumerate(triples[idx_1 + 1:]):
                if cond(candidate_t1, candidate_t2):
                    triple2 = triples.pop(idx_1 + 1 + idx_2)
                    break
            if triple2:
                triple1 = triples.pop(idx_1)
                break
        if not triple2:
            raise Exception('Ran out of number triples')

        return triple1, triple2


    def gen_numbers(self, eq, n_vars, max_n, n_changed_vars='one', same_result=False):
        if same_result:
            n_changed_vars = "two"
        op, num0, num1 = eq.split(' ')
        i = int(num0.replace('number', ''))
        j = int(num1.replace('number', ''))

        vars1 = [random.randint(1, max_n) for _ in range(n_vars)]

        triples = self.triples_dict[op]

        if same_result:
            triple1, triple2 = self.get_two_triples_with_condition('same_result', triples)

        elif n_changed_vars == 'one':
            triple1, triple2 = self.get_two_triples_with_condition('one_number_fixed', triples)

        else:
            triple1, triple2 = self.get_two_triples_with_condition('both_numbers_change', triples)

        if n_changed_vars == 'one' or n_changed_vars == 'two':
            vars2 = vars1.copy()
        elif n_changed_vars == 'all':
            vars2 = [random.randint(1, max_n) for _ in range(n_vars)]
        else:
            raise Exception('n_changed_vars not recognized: {}'.format(n_changed_vars))

        self.check_correctness(eq, eq, triple1, triple2, max_n)

        x1, y1, res1 = triple1
        x2, y2, res2 = triple2

        vars1[i] = x1
        vars1[j] = y1
        vars2[i] = x2
        vars2[j] = y2

        instance = {'vars_base': vars1, 'vars_alt': vars2}

        if same_result:
            assert res1 == res2
            assert x1 != x2 and y1 != y2
            instance['res'] = res1
        else:
            assert res1 != res2, f'{triple1} vs {triple2}'
            if n_changed_vars == 'one':
                assert x1 != x2 or y1 != y2, f'{triple1} vs {triple2}'
            else:
                assert x1 != x2 and y1 != y2, f'{triple1} vs {triple2}'
            instance['res_base'] = res1
            instance['res_alt'] = res2

        return instance


    def gen_numbers_different_result(self, eq1, eq2, n_vars, max_n):
        vars = [random.randint(1, max_n) for _ in range(n_vars)]
        op1, num0, num1 = eq1.split(' ')
        op2, num0_alt, num1_alt = eq2.split(' ')
        assert num0 == num0_alt
        assert num1 == num1_alt

        i = int(num0.replace('number', ''))
        j = int(num1.replace('number', ''))

        triples_op1 = self.triples_dict[op1]

        triple1 = None
        triple2 = None

        for idx_1, candidate_t1 in enumerate(triples_op1):
            a, b, temp_res = candidate_t1
            possible_res2 = self.pairs_dict[(a,b)][op2]
            if possible_res2:
                triple2 = (a, b, possible_res2)
                triple1 = triples_op1.pop(idx_1)
                break
        if not triple2:
            raise Exception('Ran out of number triples')

        x, y, res1 = triple1
        _, _, res2 = triple2

        self.check_correctness(eq1, eq2, triple1, triple2, max_n)
        assert res1 != res2, f'{triple1} vs {triple2}'

        vars[i] = x
        vars[j] = y

        instance = {'vars': vars, 'res_base': res1, 'res_alt' : res2}

        return instance










