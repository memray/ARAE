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

from regularizer import calc_gradient_penalty

args = run_cad.load_cadgan_args()

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
logger = init_logger(os.path.join(args.save, "log.txt"))
logger.info(vars(args))
logger.info("Vocabulary Size: char vocab={}, word vocab={}".format(len(char_word2idx), len(word_word2idx)))

# exp dir
create_exp_dir(os.path.join(args.save), ['train.py', 'models.py', 'utils.py'],
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


gan_gen = MLP_G(input_dim=args.z_size, output_dim=args.nhidden, arch_layers=args.arch_g)
gan_disc = MLP_D(input_dim=args.nhidden, output_dim=1, arch_layers=args.arch_d)

print(char_ae)
print(word_ae)
print(gan_gen)
print(gan_disc)

optimizer_ae = optim.SGD(char_ae.parameters() + word_ae.parameters(), lr=args.lr_ae)
optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))

if torch.cuda.is_available():
    char_ae = char_ae.cuda()
    word_ae = word_ae.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()


###############################################################################
# Training code
###############################################################################

def evaluate_autoencoder(autoencoder, data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        source = Variable(source.cuda(), volatile=True)
        target = Variable(target.cuda(), volatile=True)

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

        aeoutf = os.path.join(args.save, "autoencoder.txt")
        with open(aeoutf, "a") as f:
            max_values, max_indices = torch.max(output, 2)
            max_indices = \
                max_indices.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            for t, idx in zip(target, max_indices):
                # real sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f.write(chars + '\n')
                # autoencoder output sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
                f.write(chars + '\n'*2)

    return total_loss.item() / len(data_source), all_accuracies/bcnt


def gen_fixed_noise(noise, to_save):
    gan_gen.eval()
    autoencoder.eval()

    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)

    with open(to_save, "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars + '\n')


def train_lm():
    save_path = os.path.join("/tmp", ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(6)))

    indices = []
    noise = Variable(torch.ones(100, args.z_size).cuda())
    for i in range(1000):
        noise.data.normal_(0, 1)
        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)
        indices.append(max_indices.data.cpu().numpy())
    indices = np.concatenate(indices, axis=0)

    with open(save_path, "w") as f:
        # laplacian smoothing
        for word in corpus.dictionary.word2idx.keys():
            f.write(word+'\n')
        for idx in indices:
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars+'\n')
    # reverse ppl
    try:
        rev_lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                            data_path=save_path,
                            output_path=save_path+".arpa",
                            N=args.N)
        with open(os.path.join(args.data_path, 'test.txt'), 'r') as f:
            lines = f.readlines()
        if args.lowercase:
            lines = list(map(lambda x: x.lower(), lines))
        sentences = [l.replace('\n', '') for l in lines]
        rev_ppl = get_ppl(rev_lm, sentences)
    except:
        print("reverse ppl error: it maybe the generated files aren't valid to obtain an LM")
        rev_ppl = 1e15
    # forward ppl
    for_lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=os.path.join(args.data_path, 'train.txt'),
                        output_path=save_path+".arpa",
                        N=args.N)
    with open(save_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    for_ppl = get_ppl(for_lm, sentences)
    return rev_ppl, for_ppl


def train_ae(epoch, batch, total_loss_ae, start_time, i):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = Variable(source.cuda())
    target = Variable(target.cuda())
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
        logger.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'.format(
                epoch, i, len(train_data),
                elapsed * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss), accuracy))
        total_loss_ae = 0
        start_time = time.time()
    return total_loss_ae, start_time


def train_gan_g():
    gan_gen.train()
    optimizer_gan_g.zero_grad()

    z = Variable(torch.Tensor(args.batch_size, args.z_size).normal_(0, 1).cuda())
    fake_hidden = gan_gen(z)
    errG = gan_disc(fake_hidden)
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def grad_hook(grad):
    #gan_norm = torch.norm(grad, p=2, dim=1).detach().data.mean()
    #print(gan_norm, autoencoder.grad_norm)
    return grad * args.grad_lambda


def train_gan_d(batch):
    gan_disc.train()
    optimizer_gan_d.zero_grad()

    # + samples
    source, target, lengths = batch
    source = Variable(source.cuda())
    target = Variable(target.cuda())
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
    errD_real = gan_disc(real_hidden.detach())
    errD_real.backward(one)

    # - samples
    z = Variable(torch.Tensor(args.batch_size, args.z_size).normal_(0, 1).cuda())
    fake_hidden = gan_gen(z)
    errD_fake = gan_disc(fake_hidden.detach())
    errD_fake.backward(mone)

    # gradient penalty
    gradient_penalty = calc_gradient_penalty(gan_disc, real_hidden.data, fake_hidden.data)
    gradient_penalty.backward()

    optimizer_gan_d.step()
    return -(errD_real - errD_fake), errD_real, errD_fake


def train_gan_d_into_ae(batch):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = Variable(source.cuda())
    target = Variable(target.cuda())
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(grad_hook)
    errD_real = gan_disc(real_hidden)
    errD_real.backward(mone)
    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip)

    optimizer_ae.step()
    return errD_real


def train():
    logger.info("Training")
    # gan: preparation
    if args.niters_gan_schedule != "":
        gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
    else:
        gan_schedule = []
    niter_gan = 1
    fixed_noise = Variable(torch.ones(args.batch_size, args.z_size).normal_(0, 1).cuda())

    best_rev_ppl = None
    impatience = 0
    for epoch in range(1, args.epochs+1):
        train_data = corpus.batchify(corpus.train, args.batch_size, shuffle=True)
        train_data = corpus.add_fake_labels(train_data, field="long")

        test_data = corpus.batchify(corpus.test, args.batch_size, shuffle=False)
        test_data = corpus.add_fake_labels(test_data, field="long")

        # update gan training schedule
        if epoch in gan_schedule:
            niter_gan += 1
            logger.info("GAN training loop schedule: {}".format(niter_gan))

        total_loss_ae = 0
        epoch_start_time = time.time()
        start_time = time.time()
        niter = 0
        niter_g = 1

        while niter < len(train_data):
            # train ae
            for i in range(args.niters_ae):
                if niter >= len(train_data):
                    break  # end of epoch
                total_loss_ae, start_time = train_ae(epoch, train_data[niter],
                                total_loss_ae, start_time, niter)
                niter += 1
            # train gan
            for k in range(niter_gan):
                for i in range(args.niters_gan_d):
                    errD, errD_real, errD_fake = train_gan_d(
                            train_data[random.randint(0, len(train_data)-1)])
                for i in range(args.niters_gan_ae):
                    train_gan_d_into_ae(train_data[random.randint(0, len(train_data)-1)])
                for i in range(args.niters_gan_g):
                    errG = train_gan_g()

            niter_g += 1
            if niter_g % 100 == 0:
                autoencoder.noise_anneal(args.noise_anneal)
                logger.info('[{}/{}][{}/{}] Loss_D: {:.8f} (Loss_D_real: {:.8f} '
                        'Loss_D_fake: {:.8f}) Loss_G: {:.8f}'.format(
                         epoch, args.epochs, niter, len(train_data),
                         errD.data.item(), errD_real.data.item(),
                         errD_fake.data.item(), errG.data.item()))
        # eval
        test_loss, accuracy = evaluate_autoencoder(test_data, epoch)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                'test ppl {:5.2f} | acc {:3.3f}'.format(epoch,
                (time.time() - epoch_start_time), test_loss,
                math.exp(test_loss), accuracy))
        gen_fixed_noise(fixed_noise, os.path.join(args.save,
                "{:03d}_examplar_gen".format(epoch)))

        # eval with rev_ppl and for_ppl
        rev_ppl, for_ppl = train_lm(args.data_path)
        logger.info("Epoch {:03d}, Reverse perplexity {}".format(epoch, rev_ppl))
        logger.info("Epoch {:03d}, Forward perplexity {}".format(epoch, for_ppl))
        if best_rev_ppl is None or rev_ppl < best_rev_ppl:
            impatience = 0
            best_rev_ppl = rev_ppl
            logger.info("New saving model: epoch {:03d}.".format(epoch))
            save_model()
        else:
            if not args.no_earlystopping and epoch >= args.min_epochs:
                impatience += 1
                if impatience > args.patience:
                    logger.info("Ending training")
                    sys.exit()

if __name__ == '__main__':
    train()

