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
        self.tag = "sg"

    def vI_out(self, x_lookup, word_image):
        y = self.WI(x_lookup)
        return [y]

    def forward(self, x, y, word_image):
        x_lookup, y_lookup, neg_lookup, word_image = self.prepare_inputs(
            x, y, word_image
        )
        vO = self.WO(y_lookup)
        samples = self.WO(neg_lookup)
        out = self.vI_out(x_lookup, word_image)
        vI = out[0]

        pos_score = F.logsigmoid(t.dot(vO, vI))
        neg_score = F.logsigmoid(-t.mv(samples, vI))

        loss = -pos_score - t.sum(neg_score)
        return loss

