import os
import json
import codecs
import argparse
import re

"""
Transforms CAD data into lines of text files
    (data format required for ARAE model).
"""


def transform_data(in_path):
    print("Loading", in_path)

    premises = []
    hypotheses = []

    last_premise = None
    with codecs.open(in_path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)

            # load premise
            raw_premise = loaded_example['sentence1_binary_parse'].split(" ")
            premise_words = []
            # loop through words of premise binary parse
            for word in raw_premise:
                # don't add parse brackets
                if word != "(" and word != ")":
                    premise_words.append(word)
            premise = " ".join(premise_words)

            # load hypothesis
            raw_hypothesis = \
                loaded_example['sentence2_binary_parse'].split(" ")
            hypothesis_words = []
            for word in raw_hypothesis:
                if word != "(" and word != ")":
                    hypothesis_words.append(word)
            hypothesis = " ".join(hypothesis_words)

            # make sure to not repeat premiess
            if premise != last_premise:
                premises.append(premise)
            hypotheses.append(hypothesis)

            last_premise = premise

    return premises, hypotheses


def write_sentences(write_path, premises, hypotheses, append=False):
    print("Writing {} examples to {}\n".format(len(premises)+len(hypotheses), write_path))
    if append:
        with open(write_path, "a") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')
    else:
        with open(write_path, "w") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')


def load_sf_lm_from_csv(in_file_path, sf_col, lf_col, cui_col=None, trim_brackets=False):
    examples = []
    for line in open(in_file_path, 'r'):
        items = line.split('|')
        if trim_brackets and \
                (re.findall(r'\(.*?\)', items[lf_col]) or
                 re.findall(r'\[.*?\]', items[lf_col]) or
                 re.findall(r'\{.*?\}', items[lf_col])):
            print('-' * 50)
            print(items[sf_col].strip())
            print(len(items[lf_col]))
            print(items[lf_col])

            trimmed_lf = items[lf_col]
            trimmed_lf = re.sub(r'\(.*?\)', '', trimmed_lf)
            trimmed_lf = re.sub(r'\[.*?\]', '', trimmed_lf)
            trimmed_lf = re.sub(r'\{.*?\}', '', trimmed_lf)
            trimmed_lf = trimmed_lf.strip()

            print(len(trimmed_lf))
            print(trimmed_lf)
            print()
        else:
            trimmed_lf = items[lf_col].strip()

        e = {
            'cui': items[cui_col].strip() if cui_col else '',
            'short': items[sf_col].strip(),
            'ori_short': '',
            'long': trimmed_lf,
            'context': '',
            'trunc_context': '',
        }
        examples.append(e)

    return examples


def load_mimic_data(in_file, context_window):
    examples = []
    line_count = 0

    for line in open(in_file, 'r'):
        line_count += 1
        line = line.replace('name-deid', '_%#NAME#%_')
        line = line.replace('date-deid', '_%#DDMM#%_')
        tokens = line.split(' ')

        target_info = []

        # find each target and put its short form back to text to avoid info leaking
        for i in range(len(tokens)):
            if tokens[i].startswith('abbr|'):
                target_info.append((i, tokens[i]))
                items = tokens[i].split('|')
                tokens[i] = items[1].strip()

        for t_id, t_info in target_info:
            items = t_info.split('|')
            sf = items[1].strip()
            cui = items[2].strip()
            lf = ' '.join(items[3].strip().split('_'))

            pre_context = tokens[: t_id]
            post_context = tokens[t_id + 1: ]
            full_context = ' '.join(pre_context + ['<target>'] + post_context)
            pre_context = pre_context[max(t_id-context_window, 0): t_id]
            post_context = post_context[: min(len(post_context), context_window)]
            trunc_context = ' '.join(pre_context + ['<target>'] + post_context)

            e = {
                'cui': cui,
                'short': sf,
                'ori_short': "",
                'long': lf,
                'context': "",
                'trunc_context': trunc_context,
            }

            examples.append(e)

        if line_count % 1000 == 0:
            print(line_count)

    return examples

def load_umn_data(in_file, context_window):

    def _tokenize(text):
        text = re.sub(r'[\+\-\*\/<>,\(\)\.\'\"]', ' \g<0> ', text)
        tokens = text.split(' ')

        return tokens

    examples = []
    for line in open(in_file, 'rb'):
        line = line.decode(encoding='utf-8', errors='ignore')
        items = line.split('|')
        sf = items[0].strip()
        lf = items[1].strip()
        ori_sf = items[2].strip()
        sf_start = int(items[3].strip())
        sf_end = int(items[4].strip())
        context = items[6].strip()

        # insert <t> </t> before and after the target abbr/acr
        pre_context = _tokenize(context[:sf_start])
        post_context = _tokenize(context[sf_end + 1:])
        full_context = ' '.join(pre_context + ['<target>'] + post_context)

        if ori_sf != context[sf_start: sf_end + 1]:
            print('-' * 50)
            print(line)
            print(ori_sf)
            print(context[sf_start: sf_end + 1])


        # truncate context according to the number of words
        if len(pre_context) > context_window:
            pre_context = pre_context[len(pre_context) - context_window: ]
        if len(post_context) > context_window:
            post_context = post_context[: context_window]

        truncated_context = ' '.join(pre_context + ['<target>'] + post_context)

        e = {
            'cui': '',
            'short': sf,
            'ori_short': ori_sf,
            'long': lf,
            'context': full_context,
            'trunc_context': truncated_context,
        }
        examples.append(e)

    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="umls",
                        help='Name of the dataset, currently supporting mimic/umls/umn')
    parser.add_argument('--in_path', type=str, default="data/umls/",
                        help='path to data file')
    parser.add_argument('--out_path', type=str, default="data/umls/",
                        help='path to write processed data to')
    parser.add_argument('--context_window', '-c_win', type=int, default=50,
                        help='Number of words before and after the target ACR/ABBR to be used as context. '
                             'Per Sungrim Moon\'s results, 40 is a good number.')
    args = parser.parse_args()

    assert args.dataset in ['mimic', 'umls', 'umn', 'mimic_small']
    args.in_path = "data/%s/" % args.dataset
    args.out_path = "data/%s/" % args.dataset

    if args.dataset == 'umls':
        in_file = os.path.join(args.in_path, 'LRABR')
        examples = load_sf_lm_from_csv(in_file, sf_col=1, lf_col=4)

        test_examples = [e for id, e in enumerate(examples) if id % 100 == 0]
        train_examples = [e for id, e in enumerate(examples) if id % 100 != 0]
        # write to file, each line is a data example in json type
        write_sentences(write_path=os.path.join(args.out_path, 'train_no_context.txt'),
                        premises=[json.dumps(d) for d in train_examples], hypotheses=[])
        write_sentences(write_path=os.path.join(args.out_path, 'test_no_context.txt'),
                        premises=[json.dumps(d) for d in test_examples], hypotheses=[])
    elif args.dataset == 'mimic_small':
        in_file = os.path.join(args.in_path, 'train')
        examples = load_mimic_data(in_file, context_window=args.context_window)
        write_sentences(write_path=os.path.join(args.out_path, 'train.txt'),
                        premises=[json.dumps(d) for d in examples], hypotheses=[])
        in_file = os.path.join(args.in_path, 'eval')
        examples = load_mimic_data(in_file, context_window=args.context_window)
        write_sentences(write_path=os.path.join(args.out_path, 'test.txt'),
                        premises=[json.dumps(d) for d in examples], hypotheses=[])
    elif args.dataset == 'mimic':
        in_file = os.path.join(args.in_path, 'train')
        examples = load_mimic_data(in_file, context_window=args.context_window)
        write_sentences(write_path=os.path.join(args.out_path, 'train.txt'),
                        premises=[json.dumps(d) for d in examples], hypotheses=[])
        in_file = os.path.join(args.in_path, 'eval')
        examples = load_mimic_data(in_file, context_window=args.context_window)
        write_sentences(write_path=os.path.join(args.out_path, 'test.txt'),
                        premises=[json.dumps(d) for d in examples], hypotheses=[])
    elif args.dataset == 'umn':
        in_file = os.path.join(args.in_path, 'sense_inventory_1.txt')
        train_examples = load_sf_lm_from_csv(in_file, sf_col=0, lf_col=1, cui_col=5, trim_brackets=True)

        # use no-context examples for train
        write_sentences(write_path=os.path.join(args.out_path, 'train_no_context.txt'),
                        premises=[json.dumps(d) for d in train_examples], hypotheses=[])

        in_file = os.path.join(args.in_path, 'data.txt')
        examples = load_umn_data(in_file, context_window=args.context_window)
        test_examples = [e for id, e in enumerate(examples) if id % 50 == 0]
        train_examples = [e for id, e in enumerate(examples) if id % 50 != 0]
        # context examples for train/test
        write_sentences(write_path=os.path.join(args.out_path, 'train.txt'),
                        premises=[json.dumps(d) for d in train_examples], hypotheses=[])
        write_sentences(write_path=os.path.join(args.out_path, 'test.txt'),
                        premises=[json.dumps(d) for d in test_examples], hypotheses=[])
