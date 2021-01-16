import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class VAT(nn.Module):

    def __init__(self, model):
        super(VAT, self).__init__()
        self.model = model
        self.n_power = 1
        self.XI = 0.001
        self.epsilon = 0.5

    def forward(self, X, logit):
        vat_loss = virtual_adversarial_loss(X, logit, self.model, self.n_power, self.XI, self.epsilon)
        return vat_loss


def calculate_dist(x, y):
    euclidean_distance = torch.sum(F.pairwise_distance(x, y))
    return euclidean_distance


'''
d.shape = 32x1x28x28
d_reshaped.shape = 32x784x1x1
torch.norm(d_reshaped, dim=1, keepdim=True).shape = 32x1x1x1
'''


def l2_normalize(r):
    r_reshaped = r.view(r.shape[0], -1, *(1 for _ in range(r.dim() - 2)))
    r /= (torch.norm(r_reshaped, dim=1, keepdim=True) + 1e-8)
    return r


def generate_virtual_adversarial_perturbation(x, output, model, n_power, XI, epsilon):
    r = torch.normal(0, epsilon / np.sqrt(x.dim()), x.shape).cuda()

    for _ in range(n_power):
        r = r.requires_grad_()
        output1 = model(x + r)
        dist = calculate_dist(output, output1)
        grad = torch.autograd.grad(dist, [r])[0]
        r = grad.detach()

    return epsilon * l2_normalize(r)


def virtual_adversarial_loss(x, output, model, n_power, XI, epsilon):
    r_adv = generate_virtual_adversarial_perturbation(x, output, model, n_power, XI, epsilon)
    output1 = output.detach()
    output2 = model(x + r_adv)
    loss = calculate_dist(output1, output2)
    return loss
