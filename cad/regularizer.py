# -*- coding: utf-8 -*-
"""
from https://github.com/ritheshkumar95/energy_based_generative_models/blob/master/networks/regularizers.py
"""

import torch

def score_penalty(netE, real_data, beta=1.):
    '''
    From Maximum Entropy Generators for Energy-Based Models (https://arxiv.org/pdf/1901.08508.pdf)
    :param netE:
    :param real_data:
    :param beta:
    :return:
    '''
    real_data.requires_grad_(True)
    energy = netE(real_data) * beta
    score = torch.autograd.grad(
        outputs=energy, inputs=real_data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return (score.norm(2, dim=1) ** 2).mean()


def gradient_penalty(netD, real_data, fake_data):
    '''
    WGAN-GP
    :param netD:
    :param real_data: a triple (short_encoding, long_encoding, context_encoding)
    :param fake_data: a triple (short_encoding, fake_long_encoding, context_encoding)
    :return:
    '''
    real_long_encoding = real_data[1]
    fake_long_encoding = fake_data[1]
    alpha = torch.rand_like(real_long_encoding)
    interpolate_long_encoding = alpha * real_long_encoding + (1 - alpha) * fake_long_encoding
    interpolate_long_encoding.requires_grad_(True)
    real_data[0].requires_grad_(True)
    real_data[2].requires_grad_(True)

    interpolates = (real_data[0], interpolate_long_encoding, real_data[2])

    disc_interpolates = netD(*interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_gradient_penalty(netD, real_data, fake_data, gp_weight=10.0):
    ''' Steal from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py '''
    bsz = real_data.size(0)
    alpha = torch.rand(bsz, 1)
    alpha = alpha.expand(bsz, real_data.size(1))  # only works for 2D XXX
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_weight
    return gradient_penalty
