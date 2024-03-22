import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
import copy

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        """
        :rtype: object
        """
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


class MutualEnsemble(torch.nn.Module):
    def __init__(self, model_list):
        super(MutualEnsemble, self).__init__()
        self.models = model_list

    # x is batch
    def forward(self, **kwargs):
        # hard code here, to be improved
        print(f"the input x looks like {kwargs}")
        logits_total = torch.zeros(32, 8)
        for i in range(len(self.models)):
            logit = self.models[i](**kwargs).logits
            logits_total += logit
        logits_e = logits_total / len(self.models)

        return logits_e
    