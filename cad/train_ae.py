import argparse
import os
import time
import math
import numpy as np
import random
import sys
import shutil
import json
import string

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import run_cad
from utils import to_gpu, LMCorpus, train_ngram_lm, get_ppl, create_exp_dir, save_ckpt
from models import Seq2Seq, MLP_D, MLP_G


args = run_cad.load_lm_args()

# Set the random seed manually for reproducibility.
random.seed(args.seed) 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
# create corpus
corpus = LMCorpus(args.data_path,
                maxlen=args.maxlen,
                fields=args.fields,
                token_level=args.token_level,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                cut_by_cnt=False)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens

# exp dir
create_exp_dir(os.path.join(args.save), ['train_ae.py', 'models.py', 'utils.py'],
        dict=corpus.dictionary.word2idx, options=args)

def logging(str, to_stdout=True):
    with open(os.path.join(args.save, 'log.txt'), 'a') as f:
        f.write(str + '\n')
    if to_stdout:
        print(str)
logging(str(vars(args)))

eval_batch_size = 32
test_data = corpus.batchify(corpus.test, eval_batch_size, shuffle=False)
train_data = corpus.batchify(corpus.train, args.batch_size, shuffle=True)

print("Loaded data!")
print("Training data! \t: %d examples, %d batches" % (len(corpus.train), len(train_data)))
print("Test data! \t: %d examples, %d batches" % (len(corpus.test), len(test_data)))

###############################################################################
# Build the models
###############################################################################
autoencoder = Seq2Seq(emsize=args.emsize,
                      nhidden=args.nhidden,
                      ntokens=args.ntokens,
                      nlayers=args.nlayers,
                      noise_r=args.noise_r,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout)

print(autoencoder)

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)

if torch.cuda.is_available():
    autoencoder = autoencoder.cuda()
    one = torch.Tensor(1).fill_(1).cuda()
else:
    one = torch.Tensor(1).fill_(1)

# global vars
mone = one * -1

###############################################################################
# Training code
###############################################################################

def evaluate_autoencoder(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0

    aeout_path = os.path.join(args.save, "autoencoder_epoch%d.txt" % epoch)
    output_file = open(aeout_path, "w+")

    for i, batch in enumerate(data_source):
        # print("validate batch %d" % i)
        source, target, lengths = batch

        if torch.cuda.is_available():
            source = Variable(source).cuda()
            target = Variable(target).cuda()
        else:
            source = Variable(source)
            target = Variable(target)

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=True)
        flattened_output = output.view(-1, ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += F.cross_entropy(masked_output, masked_target).data

        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies += \
            torch.mean(max_indices.eq(masked_target).float()).data.item()
        bcnt += 1

        # write to file
        max_values, max_indices = torch.max(output, 2)
        max_indices = \
            max_indices.view(output.size(0), -1).data.cpu().numpy()
        target = target.view(output.size(0), -1).data.cpu().numpy()
        for t, idx in zip(target, max_indices):
            # real sentence
            chars = " ".join([corpus.dictionary.idx2word[x] for x in t
                              if corpus.dictionary.idx2word[x] != '<pad>'])
            output_file.write(chars + '\n')
            # autoencoder output sentence
            chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
            output_file.write(chars + '\n'*2)

    output_file.close()

    return total_loss.item() / len(data_source), all_accuracies/bcnt


def train_ae(epoch, batch, total_loss_ae, start_time, i):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch

    if torch.cuda.is_available():
        source = Variable(source).cuda()
        target = Variable(target).cuda()
    else:
        source = Variable(source)
        target = Variable(target)

    # print(source.shape)

    output = autoencoder(source, lengths, noise=True)

    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
    flat_output = output.view(-1, ntokens)
    masked_output = flat_output.masked_select(output_mask).view(-1, ntokens)
    loss = F.cross_entropy(masked_output, masked_target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    total_loss_ae += loss.data.item()
    if i % args.log_interval == 0:
        probs = F.softmax(masked_output, dim=-1)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data.item()
        cur_loss = total_loss_ae / args.log_interval
        elapsed = time.time() - start_time
        logging('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'.format(
                epoch, i, len(train_data),
                elapsed * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss), accuracy))
        total_loss_ae = 0
        start_time = time.time()
    return total_loss_ae, start_time


def train():
    logging("Training text AE")

    # gan: preparation
    if args.niters_gan_schedule != "":
        gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
    else:
        gan_schedule = []
    niter_gan = 1

    best_test_loss = None
    impatience = 0
    for epoch in range(1, args.epochs+1):
        # update gan training schedule
        if epoch in gan_schedule:
            niter_gan += 1
            logging("GAN training loop schedule: {}".format(niter_gan))

        total_loss_ae = 0
        epoch_start_time = time.time()
        start_time = time.time()
        niter = 0

        # train ae
        for i in range(len(train_data)):
            # print("train batch %d" % i)
            total_loss_ae, start_time = train_ae(epoch, train_data[niter],
                            total_loss_ae, start_time, niter)
            niter += 1

            if niter % 10 == 0:
                autoencoder.noise_anneal(args.noise_anneal)
                logging('[{}/{}][{}/{}]'.format(
                         epoch, args.epochs, niter, len(train_data)))
        # eval
        test_loss, accuracy = evaluate_autoencoder(test_data, epoch)
        logging('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                'test ppl {:5.2f} | acc {:3.3f}'.format(epoch,
                (time.time() - epoch_start_time), test_loss,
                math.exp(test_loss), accuracy))

        save_ckpt("ckpt_epoch%d" % epoch, args.save, autoencoder, args, corpus)
        if best_test_loss is None or test_loss < best_test_loss:
            impatience = 0
            best_test_loss = test_loss
            logging("New saving model: epoch {}. best valid score={.6f}".format(epoch, best_test_loss))
            save_ckpt("ckpt_epoch%d-best@%.6f" % (epoch, best_test_loss),
                       args.save, autoencoder, args, corpus)
        else:
            if not args.no_earlystopping and epoch >= args.min_epochs:
                impatience += 1
                if impatience > args.patience:
                    logging("Ending training")
                    sys.exit()

if __name__ == '__main__':
    train()
