import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from torch.autograd import Variable


class MyNet(nn.Module):
    def __init__(
        self,
        embed_size=75,
        vocab_size=10000,
        neg_dist=None,
        neg_samples=5,
        get_image=None
    ):
        super(MyNet, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.neg_samples = neg_samples
        self.neg_dist = neg_dist
        self.get_image = get_image

        init_width = 0.5 / embed_size
        x = [
            np.random.uniform(-init_width, init_width, (vocab_size, embed_size))
            for i in range(2)
        ]

        self.WI = nn.Embedding(vocab_size, embed_size, sparse=True).double()
        self.WI.weight.data.uniform_(-init_width, init_width)
        self.WO = nn.Embedding(vocab_size, embed_size, sparse=True).double()
        self.WO.weight.data.uniform_(-init_width, init_width)

        n_filters = 10

        self.cnn = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=5, stride=1, padding=2).double(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.fc1 = nn.Linear(640, embed_size).double()
        self.fc2 = nn.Linear(embed_size, embed_size).double()
        self.T = nn.Parameter(t.rand(embed_size), requires_grad=True).double()
        self.alpha = nn.Parameter(t.tensor([1.0]), requires_grad=True).double()
        self.beta = nn.Parameter(t.tensor([1.0]), requires_grad=True).double()

    def prepare_inputs(self, x, y):
        # image = self.get_image(x)
        # word_image = t.tensor(image, dtype=t.double)
        y_lookup = t.tensor(y, dtype=t.long)
        x_lookup = t.tensor(x, dtype=t.long)
        neg_indexes = np.random.randint(0, len(self.neg_dist), size=self.neg_samples)
        neg_indexes = self.neg_dist[neg_indexes]
        neg_lookup = t.tensor(neg_indexes, dtype=t.long)
        return x_lookup, y_lookup, neg_lookup

    def get_embedding(self, x):
        # image = self.get_image(x)
        # word_image = t.tensor(image, dtype=t.double)
        # x_lookup = t.tensor(x, dtype=t.long)
        out = self.vI_out(x)
        result = [r.detach().numpy() for r in out]
        return result

    def save_embedding(self, embed_dict, file_name):
        file = open(file_name, encoding="utf8", mode="w")
        file.write("{0} {1}\n".format(len(embed_dict), self.embed_size))
        for word in embed_dict.keys():
            e = embed_dict[word]
            e = " ".join([str(x) for x in e])
            file.write("{0} {1}\n".format(word, e))
        file.close()

