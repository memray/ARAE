# -*- coding: utf-8 -*-
"""
from https://github.com/ritheshkumar95/energy_based_generative_models/blob/master/networks/regularizers.py
"""

import torch


def gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand_like(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def score_penalty(netE, real_data, beta=1.):
    real_data.requires_grad_(True)
    energy = netE(real_data) * beta
    score = torch.autograd.grad(
        outputs=energy, inputs=real_data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return (score.norm(2, dim=1) ** 2).mean()