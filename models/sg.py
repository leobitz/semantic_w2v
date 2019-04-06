import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from torch.autograd import Variable
from models.model import MyNet


class SG(MyNet):
    def __init__(self, embed_size=75, vocab_size=10000, neg_dist=None, neg_samples=5):
        super(SG, self).__init__(embed_size, vocab_size, neg_dist, neg_samples)

    def vI_out(self, x_lookup, word_image, batch_size):
        # x = self.layer1(word_image)
        # x = x.view(batch_size, -1)
        # x = self.layer2(x)
        # y = self.layer3(x)
        y = self.WI(x_lookup)
        # T = self.T(y)
        # T = F.sigmoid(self.TX)
        # C = 1 - T
        # z = y * T + x * C
        return [y]

    def forward(self, x, y, word_image):
        word_image, x_lookup, y_lookup, neg_lookup = self.prepare_inputs(
            x, y, word_image
        )

        vO = self.WO(y_lookup)
        samples = self.WO(neg_lookup)
        out = self.vI_out(x_lookup, word_image, 1)
        vI = out[0]

        pos_score = F.logsigmoid(t.dot(vO, vI))
        neg_score = F.logsigmoid(t.sum(-t.mv(samples, vI)))

        loss = -pos_score - t.sum(neg_score)
        loss = t.mean(loss)
        return loss
