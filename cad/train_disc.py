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
from utils import to_gpu, CADCorpus, train_ngram_lm, get_ppl, create_exp_dir, save_ckpt, load_ckpt, Dictionary
from models import Seq2Seq, MLP_D, MLP_G

from cad.regularizer import score_penalty, gradient_penalty

def logging(str, to_stdout=True):
    with open(os.path.join(args.save, 'log.txt'), 'a') as f:
        f.write(str + '\n')
    if to_stdout:
        print(str)

class EnergyLoss(torch.nn.Module):

    def __init__(self):
        super(EnergyLoss, self).__init__()

    def forward(self, energy_pos, energy_neg, l2_reg=False, margin=1.0, objective="softplus"):
        temp = 1.0
        ml_coeff = 1.0
        l2_coeff = 1.0
        if objective == 'logsumexp':
            energy_neg_reduced = (energy_neg - energy_neg.min())
            coeff = torch.exp(-temp * energy_neg_reduced)
            norm_constant = coeff.sum() + 1e-4
            pos_loss = torch.mean(temp * energy_pos)
            neg_loss = coeff * (-1 * temp * energy_neg) / norm_constant
            loss_ml = ml_coeff * (pos_loss + neg_loss.sum())
        elif objective == 'cd':
            pos_loss = torch.mean(temp * energy_pos)
            neg_loss = -torch.mean(temp * energy_neg)
            loss_ml = ml_coeff * (pos_loss + torch.sum(neg_loss))
        elif objective == 'softplus':
            softplus = torch.nn.Softplus()
            loss_ml = ml_coeff * softplus(temp * (energy_pos - energy_neg))

        loss_total = torch.mean(loss_ml)
        if l2_reg:
            loss_total = loss_total + \
                         l2_coeff * (torch.mean(torch.pow(energy_pos, 2))
                                     + torch.mean(torch.pow(energy_neg, 2)))

        return loss_total


args = run_cad.load_d_args()

# Set the random seed manually for reproducibility.
random.seed(args.seed) 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
# load pretraiend models and vocabs
char_ae_params, char_word2idx, char_args = load_ckpt(args.char_ckpt)
word_ae_params, word_word2idx, word_args = load_ckpt(args.word_ckpt)

# create corpus
char_vocab = Dictionary()
char_vocab.load_from_word2idx(char_word2idx)
word_vocab = Dictionary()
word_vocab.load_from_word2idx(word_word2idx)

corpus = CADCorpus(args.data_path,
                   maxlen=args.maxlen,
                   char_vocab=char_vocab,
                   word_vocab=word_vocab,
                   lowercase=args.lowercase,
                   )

# save arguments
logging("Vocabulary Size: char vocab={}, word vocab={}".format(len(char_word2idx), len(word_word2idx)))

# exp dir
create_exp_dir(os.path.join(args.save), ['train_disc.py', 'models.py', 'utils.py'],
        dict=(char_word2idx, word_word2idx), options=args)

logging(str(vars(args)))


###############################################################################
# Build the models
###############################################################################

char_ae = Seq2Seq(emsize=char_args.emsize,
                nhidden=char_args.nhidden,
                ntokens=char_args.ntokens,
                nlayers=char_args.nlayers,
                noise_r=char_args.noise_r,
                hidden_init=char_args.hidden_init,
                dropout=char_args.dropout)

char_ae.load_state_dict(char_ae_params)

word_ae = Seq2Seq(emsize=word_args.emsize,
                nhidden=word_args.nhidden,
                ntokens=word_args.ntokens,
                nlayers=word_args.nlayers,
                noise_r=word_args.noise_r,
                hidden_init=word_args.hidden_init,
                dropout=word_args.dropout)

word_ae.load_state_dict(word_ae_params)
if args.finetune_ae:
    logging("AE will be fine-tuned")
    optimizer_ae = optim.SGD(char_ae.parameters() + word_ae.parameters(),
                             lr=args.lr_ae)
else:
    logging("AE will not be fine-tuned")

logging(char_ae)
logging(word_ae)

gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))
logging(gan_disc)

criterion = EnergyLoss()

# global vars
one = torch.Tensor(1).fill_(1)
mone = one * -1

if torch.cuda.is_available():
    char_ae = char_ae.cuda()
    word_ae = word_ae.cuda()
    gan_disc = gan_disc.cuda()
    one = one.cuda()
    mone = mone.cuda()
    criterion = criterion.cuda()

###############################################################################
# Training code
###############################################################################
def validate_disc(data_batches):
    # Turn on evaluation mode which disables dropout.
    char_ae.eval()
    word_ae.eval()
    gan_disc.eval()

    total_correct = 0
    total_count = 0
    total_loss = 0.0

    for i, batch in enumerate(data_batches):
        # + samples
        short_form, short_lengths = batch['short']
        long_form, long_lengths = batch['long']
        context, context_lengths = batch['context']
        flong_form, flong_lengths = batch['fake_long']

        if torch.cuda.is_available():
            short_form = short_form.cuda()
            short_lengths = short_lengths.cuda()
            long_form = long_form.cuda()
            long_lengths = long_lengths.cuda()
            context = context.cuda()
            context_lengths = context_lengths.cuda()
            flong_form = flong_form.cuda()
            flong_lengths = flong_lengths.cuda()

        short_encoding = char_ae(short_form, short_lengths, noise=False, encode_only=True)
        long_encoding = char_ae(long_form, long_lengths, noise=False, encode_only=True)
        context_encoding = word_ae(context, context_lengths, noise=False, encode_only=True)
        flong_encoding = char_ae(flong_form, flong_lengths, noise=False, encode_only=True)

        # energy of real/fake examples
        energy_pos = gan_disc(short_encoding.detach(), long_encoding.detach(), context_encoding.detach())
        energy_neg = gan_disc(short_encoding.detach(), flong_encoding.detach(), context_encoding.detach())

        total_loss += criterion(energy_pos, energy_neg).item()

        total_correct += torch.lt(energy_pos, energy_neg).sum().item()
        total_count += short_lengths.size(0)

        logging("current accuracy = %d/%d = %.6f" % (total_correct, total_count, float(total_correct)/float(total_count)))

    return total_loss/float(total_count) , float(total_correct)/float(total_count), total_correct, total_count


def train_gan_d(batch):
    char_ae.train()
    word_ae.train()
    gan_disc.train()
    optimizer_gan_d.zero_grad()

    # + samples
    short_form, short_lengths = batch['short']
    long_form, long_lengths = batch['long']
    context, context_lengths = batch['context']
    flong_form, flong_lengths = batch['fake_long']

    if torch.cuda.is_available():
        short_form = short_form.cuda()
        short_lengths = short_lengths.cuda()
        long_form = long_form.cuda()
        long_lengths = long_lengths.cuda()
        context = context.cuda()
        context_lengths = context_lengths.cuda()
        flong_form = flong_form.cuda()
        flong_lengths = flong_lengths.cuda()

    short_encoding = char_ae(short_form, short_lengths, noise=False, encode_only=True)
    long_encoding = char_ae(long_form, long_lengths, noise=False, encode_only=True)
    context_encoding = word_ae(context, context_lengths, noise=False, encode_only=True)
    flong_encoding = char_ae(flong_form, flong_lengths, noise=False, encode_only=True)

    # energy of real/fake examples
    energy_pos = gan_disc(short_encoding.detach(), long_encoding.detach(), context_encoding.detach())
    energy_neg = gan_disc(short_encoding.detach(), flong_encoding.detach(), context_encoding.detach())

    # compute the loss and back-propagate it
    energy_loss = criterion(energy_pos, energy_neg)
    energy_loss.backward()

    penalize_score = True
    penalize_gradient = True
    lamda = 10
    if penalize_score:
        penalty = score_penalty(gan_disc, real_data=(short_encoding.detach(), long_encoding.detach(), context_encoding.detach()))
        (lamda * penalty).backward()
    if penalize_gradient:
        penalty = gradient_penalty(gan_disc, real_data=(short_encoding.detach(), long_encoding.detach(), context_encoding.detach()),
                                   fake_data=(short_encoding.detach(), flong_encoding.detach(), context_encoding.detach()))
        (lamda * penalty).backward()

    optimizer_gan_d.step()

    return energy_loss, energy_pos, energy_neg


def train():
    logging("Training text AE")

    # gan: preparation
    if args.niters_gan_schedule != "":
        gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
    else:
        gan_schedule = []
    niter_gan = 1

    best_valid_acc = None
    eval_batch_size = args.batch_size

    impatience = 0
    for epoch in range(1, args.epochs+1):
        # re-batchify every epoch to shuffle the train and generate fake pairs
        train_data = corpus.batchify(corpus.train, args.batch_size, shuffle=True)
        train_data = corpus.add_fake_labels(train_data, field="long")

        test_data = corpus.batchify(corpus.test, eval_batch_size, shuffle=False)
        test_data = corpus.add_fake_labels(test_data, field="long")

        logging("Epoch %d" % epoch)
        logging("Loaded data!")
        logging("Training data! \t: %d examples, %d batches" % (len(corpus.train), len(train_data)))
        logging("Test data! \t: %d examples, %d batches" % (len(corpus.test), len(test_data)))

        # update gan training schedule
        if epoch in gan_schedule:
            niter_gan += 1
            logging("GAN training loop schedule: {}".format(niter_gan))

        total_loss, real_energy, fake_energy = 0.0, 0.0, 0.0
        epoch_start_time = time.time()
        start_time = time.time()

        # train
        for i in range(len(train_data)):
            # logging("train batch %d" % i)
            loss, errD_real, errD_fake = train_gan_d(train_data[i])
            total_loss += loss
            real_energy += errD_real.mean().item()
            fake_energy += errD_fake.mean().item()

            if i % args.log_interval == 0:
                elapsed = time.time() - start_time
                cur_loss = total_loss / args.log_interval
                logging('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                        'loss {:5.2f} | real energy {:8.2f} | fake energy {:8.2f}'.format(
                    epoch, i, len(train_data), elapsed * 1000 / args.log_interval,
                    cur_loss.item(), real_energy, fake_energy)
                )
                total_loss, real_energy, fake_energy = 0.0, 0.0, 0.0
                start_time = time.time()

        # validate
        valid_loss, valid_acc, total_correct, total_count = validate_disc(test_data)
        logging('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                'acc {:3.3f} | #(correct) = {} | #(all) = {}'
                .format(epoch, (time.time() - epoch_start_time),
                        valid_loss, valid_acc, total_correct, total_count))

        save_ckpt(ckpt_name="ckpt_epoch%d" % epoch, save_dir=args.save,
                  model_dict={"char_ae": char_ae, "word_ae": word_ae, "gan_disc": gan_disc},
                  args=args, vocab=(char_vocab.word2idx, word_vocab.word2idx))
        if best_valid_acc is None or valid_acc > best_valid_acc:
            impatience = 0
            best_valid_acc = valid_acc
            logging("New saving model: epoch {}, best acc={}.".format(epoch, best_valid_acc))
            save_ckpt(ckpt_name="ckpt_epoch%d-best@%f" % (epoch, best_valid_acc),
                      save_dir=args.save,
                      model_dict={"char_ae": char_ae, "word_ae": word_ae, "gan_disc": gan_disc},
                      args=args, vocab=(char_vocab.word2idx, word_vocab.word2idx)
                )
        else:
            logging("Epoch {}, acc={}.".format(epoch, valid_acc))

            if not args.no_earlystopping and epoch >= args.min_epochs:
                impatience += 1
                if impatience > args.patience:
                    logging("Ending training")
                    sys.exit()

if __name__ == '__main__':
    train()
