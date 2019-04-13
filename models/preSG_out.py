import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from torch.autograd import Variable
from models.model import MyNet


class PRESG_OUT(MyNet):
	def __init__(
		self,
		embed_size=75,
		vocab_size=10000,
		neg_dist=None,
		neg_samples=5,
		get_image=None,
	):
		super(PRESG_OUT, self).__init__(
			embed_size, vocab_size, neg_dist, neg_samples, get_image
		)

	def vI_out(self, x):
		x_lookup = t.tensor(x, dtype=t.long)
		vI = self.WI(x_lookup)
		# image = self.get_image(x)

		# y = self.alpha * vI + self.beta * image
		return [vI]

	def forward(self, x, y):
		x_lookup, y_lookup, neg_lookup, neg_samples = self.prepare_inputs(x, y)
		image = self.get_image(y)

		vO = self.WO(y_lookup)
		vI = self.WI(x_lookup)
		samples = self.WO(neg_lookup)

		
		neg_images = []
		for i in range(5):
			neg_image =  self.get_image(neg_samples[i])
			# neg_image = self.cnn(neg_image)
			# neg_image = neg_image.view(-1)
			# neg_image = self.fc1(neg_image)
			neg_images.append(neg_image)
		neg_images = t.stack(neg_images)
		
		samples = samples + self.T * neg_images
		vO = vO + self.T * image

		pos_score = F.logsigmoid(t.dot(vO, vI))
		neg_score = F.logsigmoid(-t.mv(samples, vI))

		loss = -pos_score - t.sum(neg_score)
		return loss

