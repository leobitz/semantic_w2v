import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from torch.autograd import Variable
from models.model import MyNet


class CNN(MyNet):
    def __init__(
        self,
        embed_size=75,
        vocab_size=10000,
        neg_dist=None,
        neg_samples=5,
        get_image=None,
    ):
        super(CNN, self).__init__(
            embed_size, vocab_size, neg_dist, neg_samples, get_image
        )

    def vI_out(self, x):
        x_lookup = t.tensor(x, dtype=t.long)
        vI = self.WI(x_lookup)
        image = self.get_image(x)
        image = self.cnn(image)
        image = image.view(-1)
        image = self.fc1(image)

        
        return [image]

    def forward(self, x, y):
        x_lookup, y_lookup, neg_lookup = self.prepare_inputs(x, y)
        image = self.get_image(x)
        image = self.cnn(image)
        image = image.view(-1)
        vI = self.fc1(image)

        vO = self.WO(y_lookup)
        # vI = self.WI(x_lookup)
        samples = self.WO(neg_lookup)

        # vI = self.alpha * vI + self.beta * image

        pos_score = F.logsigmoid(t.dot(vO, vI))
        neg_score = F.logsigmoid(-t.mv(samples, vI))

        loss = -pos_score - t.sum(neg_score)
        return loss


