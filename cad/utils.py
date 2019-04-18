import os
import string

import torch
import numpy as np
import random
import shutil
import json
import math

def load_kenlm():
    global kenlm
    import kenlm


def to_gpu(gpu, var):
    if gpu and torch.cuda.is_available():
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.word2idx['<target>'] = 4
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]

        for w_id, (w, freq) in enumerate(sorted(vocab_list, key=lambda x:x[1], reverse=True)):
            print("\t[%d] %s \t: %d" % (w_id, w, freq))

        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
            # sort to make vocabulary deterministic
            self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("Original vocab {}; Pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}


    def __len__(self):
        return len(self.word2idx)

    def load_from_idx2word(self, idx2word):
        self.idx2word = idx2word
        self.word2idx = {v: k for k, v in self.idx2word.items()}

    def load_from_word2idx(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in self.word2idx.items()}

class LMCorpus(object):
    def __init__(self, paths, maxlen, fields,
                 vocab_size=11000, min_freq=5,
                 cut_by_cnt=False, lowercase=False, token_level='word'):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.cut_by_cnt = cut_by_cnt
        self.min_freq = min_freq
        self.vocab_size = vocab_size

        self.train = []
        self.test = []
        for path in paths:
            self.train_path = os.path.join(path, 'train.txt')
            self.test_path = os.path.join(path, 'test.txt')
            if os.path.exists(self.train_path):
                self.train.extend(self.tokenize(self.train_path, fields, token_level))
            if os.path.exists(self.test_path):
                self.test.extend(self.tokenize(self.test_path, fields, token_level))

        # make the vocabulary from training set
        if token_level == 'char':
            init_char = True
        else:
            init_char = False

        self.make_vocab(self.train, init_char=init_char)

        self.train = self.vectorize(self.train)
        self.test = self.vectorize(self.test)

        print('Data size: train %d, test %d' % (len(self.train), len(self.test)))


    def make_vocab(self, sent_list, init_char=False):
        if init_char:
            max_idx = int(np.amax(list(self.dictionary.word2idx.values())))
            for char in string.printable:
                max_idx += 1
                self.dictionary.word2idx[char] = max_idx

        for sent in sent_list:
            for word in sent:
                self.dictionary.add_word(word)

        # prune the vocabulary
        if self.cut_by_cnt:
            self.dictionary.prune_vocab(k=self.min_freq, cnt=self.cut_by_cnt)
        else:
            self.dictionary.prune_vocab(k=self.vocab_size, cnt=self.cut_by_cnt)

    def tokenize(self, path, fields, token_level):
        """
        Tokenizes a text file.
        Each line is a json, values in multiple fields might be used. Split them to individual examples.
        """
        dropped = 0
        with open(path, 'r') as f:
            line_count = 0
            examples = []
            for line in f:
                line_count += 1
                json_ex = json.loads(line)
                # ignore null inputs
                text_exs = [json_ex[f] for f in fields
                            if f in json_ex and len(json_ex[f].strip())>0]

                for text_ex in text_exs:
                    if self.lowercase:
                        text_ex = text_ex.lower().strip()
                    else:
                        text_ex = text_ex.strip()

                    if token_level == 'word':
                        tokens = text_ex.split(" ")
                    else:
                        tokens = list(text_ex)

                    if self.maxlen > 0 and len(tokens) > self.maxlen:
                        dropped += 1
                        continue

                    tokens = ['<sos>'] + tokens + ['<eos>']
                    examples.append(tokens)

        print("Number of data generated from {}: {} sentences out of {} examples. {} are dropped away.".
              format(path, len(examples), line_count, dropped))
        return examples


    def vectorize(self, sent_list):
        # vectorize
        vocab = self.dictionary.word2idx
        unk_idx = vocab['<oov>']

        return_list = []
        for tokens in sent_list:
            indices = [vocab[w] if w in vocab else unk_idx for w in tokens]
            return_list.append(indices)

        return return_list


    def batchify(self, data, bsz, shuffle=False):
        if shuffle:
            random.shuffle(data)
        nbatch = len(data) // bsz
        batches = []

        for i in range(nbatch):
            # Pad batches to maximum sequence length in batch
            batch = data[i * bsz:(i + 1) * bsz]
            # subtract 1 from lengths b/c includes BOTH starts & end symbols
            lengths = [len(x) - 1 for x in batch]
            # sort items by length (decreasing)
            batch, lengths = length_sort(batch, lengths)

            # source has no end symbol
            source = [x[:-1] for x in batch]
            # target has no start symbol
            target = [x[1:] for x in batch]

            # find length to pad to
            maxlen = max(lengths)
            for x, y in zip(source, target):
                zeros = (maxlen - len(x)) * [0]
                x += zeros
                y += zeros

            source = torch.LongTensor(np.array(source))
            target = torch.LongTensor(np.array(target)).view(-1)

            batches.append((source, target, lengths))

        return batches


def save_ckpt(ckpt_name, save_dir, model_dict, args, vocab):
    print("Saving models to {}".format(os.path.join(save_dir, "%s.pt" % ckpt_name)))
    dict_to_save = {name: model.state_dict() for name, model in model_dict.items()}
    dict_to_save['args'] = args
    dict_to_save['vocab'] = vocab
    torch.save(dict_to_save,
        os.path.join(save_dir, "%s.pt" % ckpt_name))


def load_ckpt(ckpt_path):
    print('Loading models from {}'.format(ckpt_path))
    loaded_params = torch.load(ckpt_path, map_location='cpu' if not torch.cuda.is_available() else None)
    args = loaded_params.get('args')
    word2idx = loaded_params.get('vocab')

    model_params = loaded_params.get('ae')

    return model_params, word2idx, args


class CADCorpus(object):
    def __init__(self, paths, maxlen, char_vocab, word_vocab, lowercase=False):
        self.char_dict = char_vocab
        self.word_dict = word_vocab
        self.maxlen = maxlen
        self.lowercase = lowercase

        self.train = []
        self.test = []
        for path in paths:
            self.train_path = os.path.join(path, 'train.txt')
            self.test_path = os.path.join(path, 'test.txt')

            if os.path.exists(self.train_path):
                short_forms = self.tokenize(self.train_path, fields=['short'], token_level='char')
                long_forms = self.tokenize(self.train_path, fields=['long'], token_level='char')
                contexts = self.tokenize(self.train_path, fields=['trunc_context'], token_level='word')
                self.train.extend(list(zip(short_forms, long_forms, contexts)))
            if os.path.exists(self.test_path):
                short_forms = self.tokenize(self.test_path, fields=['short'], token_level='char')
                long_forms = self.tokenize(self.test_path, fields=['long'], token_level='char')
                contexts = self.tokenize(self.test_path, fields=['trunc_context'], token_level='word')
                self.test.extend(list(zip(short_forms, long_forms, contexts)))

        self.train = self.vectorize(self.train, char_vocab, word_vocab)
        self.test = self.vectorize(self.test, char_vocab, word_vocab)

        print('#(examples): train %d, test %d' % (len(self.train), len(self.test)))


    def tokenize(self, path, fields, token_level):
        """
        Tokenizes a text file.
        Each line is a json, values in multiple fields might be used. Split them to individual examples.
        """
        dropped = 0
        with open(path, 'r') as f:
            line_count = 0
            examples = []
            for line in f:
                line_count += 1
                json_ex = json.loads(line)
                # ignore null inputs
                text_exs = [json_ex[f] for f in fields
                            if f in json_ex and len(json_ex[f].strip())>0]

                for text_ex in text_exs:
                    if self.lowercase:
                        text_ex = text_ex.lower().strip()
                    else:
                        text_ex = text_ex.strip()

                    if token_level == 'word':
                        tokens = text_ex.split(" ")
                    else:
                        tokens = list(text_ex)

                    if self.maxlen > 0 and len(tokens) > self.maxlen:
                        dropped += 1
                        continue

                    tokens = ['<sos>'] + tokens + ['<eos>']
                    examples.append(tokens)

        return examples


    def vectorize(self, examples, char_vocab, word_vocab):
        """
        :param examples: a list of triples (short_form, long_form, context)
        :param char_vocab:
        :param word_vocab:
        :return:
        """
        return_examples = []
        for triple in examples:
            short_form, long_form, context = triple
            short_idx = [char_vocab.word2idx[w] if w in char_vocab.word2idx else char_vocab.word2idx['<oov>'] for w in short_form]
            long_idx = [char_vocab.word2idx[w] if w in char_vocab.word2idx else char_vocab.word2idx['<oov>'] for w in long_form]
            context_idx = [word_vocab.word2idx[w] if w in word_vocab.word2idx else word_vocab.word2idx['<oov>'] for w in context]

            return_examples.append({
                "short": short_idx,
                "long": long_idx,
                "context": context_idx,
            })

        return return_examples


    def add_fake_labels(self, batches, field):
        """
        :param examples: a dict, contains three vectorized items: short, long, context
        :param field: which field to generate fake items
        :return:
        """
        for batch in batches:
            data, lengths = batch[field]
            example_num = data.size(0)
            choices = [np.concatenate([np.arange(0, i), np.arange(i+1, example_num)]) for i in range(example_num)]
            sampled_idx = [np.random.choice(c) for c in choices]
            sampled_idx = torch.from_numpy(np.asarray(sampled_idx))

            sampled_data = torch.index_select(data, dim=0, index=sampled_idx)
            sampled_lengths = torch.index_select(lengths, dim=0, index=sampled_idx)

            batch['fake_%s' % field] = (sampled_data, sampled_lengths)

        return batches


    def batchify(self, examples, bsz, shuffle=False):
        """
        Each example is a dict containing three items: short, long, context
        Therefore we need to pad three matrices
        :param examples:
        :param bsz:
        :param shuffle:
        :return:
        """
        keys = examples[0].keys()

        if shuffle:
            random.shuffle(examples)
        nbatch = len(examples) // bsz
        batches = []

        for i in range(nbatch):
            # Pad batches to maximum sequence length in batch
            batch_examples = examples[i * bsz:(i + 1) * bsz]
            batch = {}

            for key in keys:
                data = [x[key] for x in batch_examples]
                # subtract 1 from lengths b/c includes BOTH starts & end symbols
                lengths = [len(x) for x in data]

                # do not sort here (sort items by length decreasing)
                # data, lengths = length_sort(data, lengths)

                # find length to pad to
                maxlen = max(lengths)
                for x in data:
                    zeros = (maxlen - len(x)) * [0]
                    x += zeros

                data = torch.LongTensor(np.array(data))
                lengths = torch.LongTensor(np.array(lengths))
                batch[key] = (data, lengths)

            batches.append(batch)

        return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    #
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

    load_kenlm()
    # create language model
    assert(output_path)  # captured by try..except block outside
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        nll = np.sum([- math.log(math.pow(10.0, score)) for score, _, _ in lm.full_scores(sent, bos=True, eos=False)])
        word_count = len(words)
        total_wc += word_count
        total_nll += nll
    ppl = np.exp(total_nll / total_wc)
    return ppl


def create_exp_dir(path, scripts_to_save=None, dict=None, options=None):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    # dump the dictionary
    if dict is not None:
        with open(os.path.join(path, 'vocab.json'), 'w') as f:
            json.dump(dict, f)

    # dump the args
    if options is not None:
        with open(os.path.join(path, 'options.json'), 'w') as f:
            json.dump(vars(options), f)
