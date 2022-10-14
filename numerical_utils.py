import csv
import spacy
from functools import reduce

nlp = spacy.load("en_core_web_sm")

def create_verbs_dict(csv_file_path):
    d = {}
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='|')
        for i, row in enumerate(reader):
            if len(row) < 4:
                continue
            sing = row[0]
            third_pers = row[1]
            ps = row[2]
            pp = row[3]
            d[sing] = {'third_pers': third_pers, 'ps': ps, 'pp': pp}

    return d


def read_svamp_csv(csv_file_path, operator='all'):
    data = []
    examples_kept_n = 0
    total_examples_n = 0

    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        reader.__next__()
        for i, row in enumerate(reader):

            total_examples_n += 1
            eq = row[2]

            if operator != 'all' and operator not in eq:
                continue

            if len(eq.split(' ')) > 3:
                continue

            body = row[-2]
            question = row[-1]
            text = body + question

            if 'number4' in text:
                n_vars = 5
            elif 'number3' in text:
                n_vars = 4
            elif 'number2' in text:
                n_vars = 3
            else:
                n_vars = 2

            datum = {'id': examples_kept_n, 'body' : body, 'question' : question, 'eq' : eq, 'n_vars': n_vars}
            examples_kept_n += 1
            data.append(datum)
    print(f'Using {examples_kept_n} two-unknown examples out of {total_examples_n}')
    return data


def q_to_statement(data, dict_path, include_statements=False, include_questions=False):
    v_dict = create_verbs_dict(dict_path)
    new_data = []
    for datum in data[:]:
        q = datum['question']
        if q.startswith('how many'):
            parsed = nlp(q)
            if parsed[2].pos_ != 'AUX':
                verbs = [token for token in parsed if token.pos_ == "VERB"]
                auxs = [token for token in parsed if token.pos_ == "AUX"]
                nps = [chunk for chunk in parsed.noun_chunks if chunk.text not in ['total']]

                if len(verbs) != 1 or len(auxs) != 1 or len(nps) != 2:
                    continue

                if 'ride ' in q or 'rides ' in q or 'go' in q or 'working' in q or 'swimming' in q:
                    continue

                verb = verbs[0].text
                aux = auxs[0]
                morph_features = aux.morph.to_dict()
                if aux.text in ['could', 'would', 'should', 'will', 'can', 'must']:
                    aux = aux.text + ' '
                else:
                    aux = ''
                    if 'Tense' in morph_features and morph_features['Tense'] == 'Past':
                        if verb not in v_dict:
                            continue
                        verb = v_dict[verb]['ps']
                    elif 'Number' in morph_features and morph_features['Number'] == 'Sing' and morph_features['Person'] == '3':
                        if verb not in v_dict:
                            continue
                        verb = v_dict[verb]['third_pers']

                subj = nps[1].text
                obj = nps[0].text.replace('how many', '').strip()
                rephrased = 'the number of ' + obj + ' that ' + subj + ' ' + aux + verb + ' is'
                if include_questions and include_statements:
                    datum['statement'] = rephrased
                elif include_statements:
                    datum['statement'] = rephrased
                elif include_questions:
                    pass
                else:
                    raise Exception('No prompt specified')

                new_data.append(datum)

    return new_data


def construct_numerical_templates(path_to_data, path_to_dict, statements=False, questions=False, operator='all'):
    data = read_svamp_csv(path_to_data, operator)
    data = q_to_statement(data, path_to_dict, statements, questions)
    return data


def factors(n):
    return list(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))


def convert_to_words(num):
    # Python program to print a given number in
    # words. The program handles numbers
    # from 0 to 9999

    # Credits: Mithun Kumar

    l = len(num)

    # Base cases
    if (l == 0):
        print("empty string")
        return

    if (l > 4):
        print("Length more than 4 is not supported")
        return

    # The first string is not used,
    # it is to make array indexing simple
    single_digits = ["zero", "one", "two", "three",
                    "four", "five", "six", "seven",
                    "eight", "nine"]

    # The first string is not used,
    # it is to make array indexing simple
    two_digits = ["", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen",
                "nineteen"]

    # The first two string are not used,
    # they are to make array indexing simple
    tens_multiple = ["", "", "twenty", "thirty", "forty",
                    "fifty", "sixty", "seventy", "eighty",
                    "ninety"]

    tens_power = ["hundred", "thousand"]

    # Used for debugging purpose only
    #print(num, ":", end=" ")
    res = ''

    # For single digit number
    if (l == 1):
        res += (single_digits[ord(num[0]) - 48])
        return res.strip()

    # Iterate while num is not '\0'
    x = 0
    while (x < len(num)):

        # Code path for first 2 digits
        if (l >= 3):
            if (ord(num[x]) - 48 != 0):
                res += single_digits[ord(num[x]) - 48] + ' '
                res += tens_power[l - 3] + ' '
                # here len can be 3 or 4

            l -= 1

        # Code path for last 2 digits
        else:

            # Need to explicitly handle
            # 10-19. Sum of the two digits
            # is used as index of "two_digits"
            # array of strings
            if (ord(num[x]) - 48 == 1):
                sum = (ord(num[x]) - 48 +
                    ord(num[x+1]) - 48)
                res += two_digits[sum] + ' '
                return res.strip()

            # Need to explicitly handle 20
            elif (ord(num[x]) - 48 == 2 and
                ord(num[x + 1]) - 48 == 0):
                return "twenty"


            # Rest of the two digit
            # numbers i.e., 21 to 99
            else:
                i = ord(num[x]) - 48
                if(i > 0):
                    res += tens_multiple[i] + ' '
                else:
                    print("", end="")
                x += 1
                if(ord(num[x]) - 48 != 0):
                    res += (single_digits[ord(num[x]) - 48])
        x += 1

    return res.strip()
