import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, output, target):
        loss = F.nll_loss(torch.log(output), target)
        return loss
