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
from utils import *
from models import Seq2Seq, MLP_D, MLP_G

from regularizer import score_penalty, gradient_penalty

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


args = run_cad.load_cadgan_args()
logger = init_logger(os.path.join(args.save, "exp_log.txt"))

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
if not os.path.exists(args.save):
    os.makedirs(args.save)

logger.info("Vocabulary Size: char vocab={}, word vocab={}".format(len(char_word2idx), len(word_word2idx)))

# exp dir
create_exp_dir(os.path.join(args.save), ['train_cadgan.py', 'models.py', 'utils.py'],
        dict=(char_word2idx, word_word2idx), options=args)

logger.info(str(vars(args)))


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

D = MLP_D(input_dim=args.nhidden, output_dim=1, arch_layers=args.arch_d)
G = MLP_G(input_dim=args.nhidden, output_dim=args.nhidden, noise_dim=args.z_size, arch_layers=args.arch_g)
if args.finetune_ae:
    logger.info("AE will be fine-tuned")
    optimizer_D = optim.Adam(list(D.parameters()) + list(char_ae.parameters()) + list(word_ae.parameters()),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))
    optimizer_G = optim.Adam(list(G.parameters()) + list(char_ae.parameters()) + list(word_ae.parameters()),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))
else:
    logger.info("AE will not be fine-tuned")
    optimizer_D = optim.Adam(D.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))
    optimizer_G = optim.Adam(G.parameters(),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))

logger.info(char_ae)
logger.info(word_ae)
logger.info(D)
logger.info(G)

if torch.cuda.is_available():
    logger.info("Running on GPU")
    char_ae = char_ae.cuda()
    word_ae = word_ae.cuda()
    D = D.cuda()
    G = G.cuda()
else:
    logger.info("Running on CPU")

###############################################################################
# Training code
###############################################################################
def validate_disc(data_batches):
    # Turn on evaluation mode which disables dropout.
    char_ae.eval()
    word_ae.eval()
    D.eval()

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
        energy_pos = D(short_encoding.detach(), long_encoding.detach(), context_encoding.detach())
        energy_neg = D(short_encoding.detach(), flong_encoding.detach(), context_encoding.detach())

        total_correct += torch.lt(energy_pos, energy_neg).sum().item()
        total_count += short_lengths.size(0)

        logger.info("current accuracy = %d/%d = %.6f" % (total_correct, total_count, float(total_correct)/float(total_count)))

    return total_loss/float(total_count) , float(total_correct)/float(total_count), total_correct, total_count


def train_GAN(batch, train_G):
    char_ae.train()
    word_ae.train()
    D.train()
    optimizer_D.zero_grad()

    # + samples
    short_form, short_lengths = batch['short']
    long_form, long_lengths = batch['long']
    context, context_lengths = batch['context']
    noise = Variable(torch.ones(args.batch_size, args.z_size).normal_(0, 1))

    if torch.cuda.is_available():
        short_form = short_form.cuda()
        short_lengths = short_lengths.cuda()
        long_form = long_form.cuda()
        long_lengths = long_lengths.cuda()
        context = context.cuda()
        context_lengths = context_lengths.cuda()
        noise = noise.cuda()

    short_encoding = char_ae(short_form, short_lengths, noise=False, encode_only=True).detach()
    long_encoding = char_ae(long_form, long_lengths, noise=False, encode_only=True).detach()
    context_encoding = word_ae(context, context_lengths, noise=False, encode_only=True).detach()
    # fake_long_encoding = char_ae(flong_form, flong_lengths, noise=False, encode_only=True)

    fake_long_encoding = G(noise, short_encoding, context_encoding)

    # energy of real/fake examples
    real_D_loss = D(short_encoding, long_encoding, context_encoding).mean()
    fake_D_loss = D(short_encoding, fake_long_encoding.detach(), context_encoding).mean()

    # compute the loss and back-propagate it
    D_loss = fake_D_loss - real_D_loss

    penalize_score = False
    penalize_gradient = True
    lamda = 10
    score_penalty_loss = 0.0
    gradient_penalty_loss = 0.0

    if penalize_score:
        score_penalty_loss = score_penalty(D, real_data=(short_encoding, long_encoding, context_encoding))
        D_loss += (lamda * score_penalty_loss)
        score_penalty_loss = score_penalty_loss.item()

    if penalize_gradient:
        gradient_penalty_loss = gradient_penalty(D,
                                   real_data=(short_encoding, long_encoding, context_encoding),
                                   fake_data=(short_encoding, fake_long_encoding.detach(), context_encoding))
        D_loss += (lamda * gradient_penalty_loss)
        gradient_penalty_loss = gradient_penalty_loss.item()

    # final disc cost
    D_loss.backward()
    optimizer_D.step()

    if train_G:
        noise = Variable(torch.ones(args.batch_size, args.z_size).normal_(0, 1))
        if torch.cuda.is_available():
            noise = noise.cuda()
        fake_long_encoding = G(noise, short_encoding, context_encoding)
        G_loss = D(short_encoding, fake_long_encoding, context_encoding).mean()
        (-G_loss).backward()
        optimizer_G.step()

    return D_loss.item(), real_D_loss.item(), fake_D_loss.item(), score_penalty_loss, gradient_penalty_loss


def train():
    # gan: preparation
    if args.niters_gan_schedule != "":
        gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
    else:
        gan_schedule = []
    niter_gan = 1

    global global_step
    global_step = 0

    best_valid_acc = None
    eval_batch_size = args.batch_size

    impatience = 0
    for epoch in range(1, args.epochs+1):
        # re-batchify every epoch to shuffle the train and generate fake pairs
        train_data = corpus.batchify(corpus.train, args.batch_size, shuffle=True)
        train_data = corpus.add_fake_labels(train_data, field="long")

        test_data = corpus.batchify(corpus.test, eval_batch_size, shuffle=False)
        test_data = corpus.add_fake_labels(test_data, field="long")

        logger.info("Epoch %d" % epoch)
        logger.info("Loaded data!")
        logger.info("Training data! \t: %d examples, %d batches" % (len(corpus.train), len(train_data)))
        logger.info("Test data! \t: %d examples, %d batches" % (len(corpus.test), len(test_data)))

        # update gan training schedule
        if epoch in gan_schedule:
            niter_gan += 1
            logger.info("GAN training loop schedule: {}".format(niter_gan))

        D_losses, w_dists, real_energy, fake_energy, SP_losses, GP_losses = [], [], [], [], [], []
        epoch_start_time = time.time()
        start_time = time.time()

        # train
        for i in range(len(train_data)):
            # update global_step here, might be used in TensorboardX later
            global_step += 1

            D_loss, real_D_loss, fake_D_loss, score_penalty_loss, gradient_penalty_loss\
                = train_GAN(train_data[i], train_G=(i % args.niters_gan_g == 0))

            D_losses.append(D_loss)
            w_dists.append((real_D_loss-fake_D_loss))
            real_energy.append(real_D_loss)
            fake_energy.append(fake_D_loss)
            SP_losses.append(score_penalty_loss)
            GP_losses.append(gradient_penalty_loss)

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                logger.info('| step {:3d} | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                            'D-loss {:5.2f} | Estimated W-dist {:5.2f} | '
                            'real energy {:8.2f} | fake energy {:8.2f} | '
                            'score penalty {:8.2f} | gradient penalty {:8.2f}'.format(
                    global_step, epoch, i, len(train_data), elapsed * 1000 / args.log_interval,
                    np.average(D_losses), np.average(w_dists),
                    np.average(real_energy), np.average(fake_energy),
                    np.average(SP_losses), np.average(GP_losses)
                )
                )
                D_losses, w_dists, real_energy, fake_energy, SP_losses, GP_losses = [], [], [], [], [], []
                start_time = time.time()

            if global_step % args.save_every == 0:
                save_ckpt(ckpt_name="ckpt_epoch%d" % epoch, save_dir=args.save,
                          model_dict={"char_ae": char_ae, "word_ae": word_ae, "D": D},
                          args=args, vocab=(char_vocab.word2idx, word_vocab.word2idx))

            if global_step % args.valid_every == 0:
                # validate
                valid_loss, valid_acc, total_correct, total_count = validate_disc(test_data)
                logger.info('| Validation {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                        'acc {:3.3f} | #(correct) = {} | #(all) = {}'
                        .format(epoch, (time.time() - epoch_start_time),
                                valid_loss, valid_acc, total_correct, total_count))

                if best_valid_acc is None or valid_acc > best_valid_acc:
                    impatience = 0
                    best_valid_acc = valid_acc
                    logger.info("New saving model: epoch {}, best acc={}.".format(epoch, best_valid_acc))
                    save_ckpt(ckpt_name="ckpt_epoch%d-best@%f" % (epoch, best_valid_acc),
                              save_dir=args.save,
                              model_dict={"char_ae": char_ae, "word_ae": word_ae, "D": D},
                              args=args, vocab=(char_vocab.word2idx, word_vocab.word2idx)
                        )
                else:
                    logger.info("Epoch {}, acc={}.".format(epoch, valid_acc))

                    if not args.no_earlystopping and epoch >= args.min_epochs:
                        impatience += 1
                        if impatience > args.patience:
                            logger.info("Ending training")
                            sys.exit()

if __name__ == '__main__':
    train()
