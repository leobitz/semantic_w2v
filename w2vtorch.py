import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from data_handle import *
from torch.autograd import Variable
import os


class Net(nn.Module):

	def __init__(self, embed_size=75,
				 vocab_size=10000,
				 neg_dist=None,
				 neg_samples=5,
				 lr=0.025,
				 device='cpu'):
		super(Net, self).__init__()
		self.embed_size = embed_size
		self.vocab_size = vocab_size
		self.neg_samples = neg_samples
		self.neg_dist = neg_dist
		self.lr = lr
		init_width = 0.5 / embed_size
		x = [np.random.uniform(-init_width, init_width,
							   (vocab_size, embed_size)) for i in range(2)]
		if device == 'gpu': device = 'cuda'
		self.device = t.device(device)

		self.WI = nn.Embedding(vocab_size, embed_size, sparse=True)
		self.WI.to(device=device, dtype=t.float64)
		self.WI.weight.data.uniform_(-init_width, init_width)
		self.WO = nn.Embedding(vocab_size, embed_size, sparse=True)
		self.WO.to(device=device, dtype=t.float64)
		self.WO.weight.data.uniform_(-init_width, init_width)
		self.alpha = nn.Parameter(t.tensor([1.0], requires_grad=True, device=device, dtype=t.float64))
		self.beta = nn.Parameter(t.tensor([1.0], requires_grad=True, device=device, dtype=t.float64))
		n_filters = 10
		if device == 'cuda':
			self.fc1 = nn.Linear(n_filters * 4 * 16, embed_size).cuda().double()
			self.layer1 = nn.Sequential(
				nn.Conv2d(1, n_filters, kernel_size=5, stride=1,
						  padding=2).cuda().double(),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3)
			)
			self.fc2 = nn.Linear(embed_size * 2, embed_size).cuda().double()
			self.fc3 = nn.Linear(embed_size, embed_size).cuda().double()
		else:
			self.fc1 = nn.Linear(n_filters * 4 * 16, embed_size).double()
			self.layer1 = nn.Sequential(
				nn.Conv2d(1, n_filters, kernel_size=5, stride=1, padding=2).double(),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3)
			)
			self.fc2 = nn.Linear(embed_size, embed_size).double()
			self.fc3 = nn.Linear(embed_size, embed_size).double()

	def vI_out(self, x_lookup, word_image, batch_size):
		input_x = self.layer1(word_image).view(batch_size, -1)
		seqI = self.fc1(input_x)
		vI = self.WI(x_lookup)
		vI = self.alpha * vI + self.beta * seqI
		# vI = ((self.alpha)/(self.alpha + self.beta)) * vI + ((self.beta)/(self.alpha + self.beta)) * seqI
		return vI

	def forward(self, word_image, x, y):
		word_image, x_lookup, y_lookup, neg_lookup = self.prepare_inputs(word_image,x, y)
		
		vO = self.WO(y_lookup)
		samples = self.WO(neg_lookup)

		vI = self.vI_out(x_lookup, word_image, len(y))

		pos_z = t.mul(vO, vI).squeeze()
		vI = vI.unsqueeze(2).view(len(x), self.embed_size, 1)
		neg_z = -t.bmm(samples, vI).squeeze()

		pos_score = t.sum(pos_z, dim=1)
		pos_score = F.logsigmoid(pos_score)
		neg_score = F.logsigmoid(neg_z)

		loss = -pos_score - t.sum(neg_score)
		loss = t.mean(loss)
		return loss

	def prepare_inputs(self, image, x, y):
		word_image = t.tensor(image, dtype=t.double, device=self.device)
		y_lookup = t.tensor(y, dtype=t.long, device=self.device)
		x_lookup = t.tensor(x, dtype=t.long, device=self.device)
		neg_indexes = np.random.randint(
			0, len(self.neg_dist), size=(len(y), self.neg_samples))  # .flatten()
		neg_indexes = self.neg_dist[neg_indexes]  # .reshape((-1, 5)).tolist()
		neg_lookup = t.tensor(neg_indexes, dtype=t.long, device=self.device)
		return word_image, x_lookup, y_lookup, neg_lookup

	def get_embedding(self, image, x):
		word_image = t.tensor(image, dtype=t.double, device=self.device)
		x_lookup = t.tensor(x, dtype=t.long, device=self.device)
		vI = self.vI_out(x_lookup, word_image, len(x))
		embeddings = vI.detach().numpy()
		vI_w = self.WI(x_lookup).detach().numpy()
		return embeddings, vI_w

	def save_embedding(self, embed_dict, file_name, device):
		file = open(file_name, encoding='utf8', mode='w')
		file.write("{0} {1}\n".format(len(word2int), self.embed_size))
		for word in embed_dict.keys():
			e = embed_dict[word]
			e = ' '.join([str(x) for x in e])
			file.write("{0} {1}\n".format(word, e))
		file.close()


def generateSG(data, skip_window, batch_size,
			   int2word, char2tup, n_chars, n_consonant, n_vowels):
	win_size = skip_window  # np.random.randint(1, skip_window + 1)
	i = win_size
	while True:
		batch_input = []
		batch_output = []
		batch_vec_input = []
		for bi in range(0, batch_size, skip_window ):
			context = data[i - win_size: i + win_size + 1]
			target = context.pop(win_size)
			context = np.random.choice(context, skip_window, replace=False)
			targets = [target] * (win_size )
			batch_input.extend(targets)
			batch_output.extend(context)

			con_mat, vow_mat = word2vec_seperated(char2tup,
												  int2word[target], n_chars, n_consonant, n_vowels)
			word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
				(1, 1, n_chars, (n_consonant + n_vowels)))
			batch_vec_input.extend([word_mat] * (win_size))
			i += 1
			if i + win_size + 1 > len(data):
				i = win_size
		batch_vec_input = np.vstack(batch_vec_input)
		yield batch_input, batch_vec_input, batch_output


words = read_file("data/news.txt")#[:2000]
words, word2freq = min_count_threshold(words)
# words = subsampling(words, 1e-3)
vocab, word2int, int2word = build_vocab(words)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
print("Words to train: ", len(words))
print("Vocabs to train: ", len(vocab))
#print("Unk count: ", word2freq['<unk>'])
int_words = words_to_ints(word2int, words)
int_words = np.array(int_words, dtype=np.int32)
n_chars = 11 + 2
n_epoch = 10
batch_size = 5
skip_window = 1
init_lr = .05
gen = generateSG(list(int_words), skip_window, batch_size,
				 int2word, char2tup, n_chars, n_consonant, n_vowel)

ns_unigrams = np.array(
	ns_sample(word2freq, word2int, int2word, .75), dtype=np.int32)
device = 'cpu'
net = Net(neg_dist=ns_unigrams, embed_size=100,
		  vocab_size=len(vocab), device=device)
sgd = optimizers.SGD(net.parameters(), lr=init_lr)
start = time.time()
losses = []
grad_time = []
forward_time = []
backward_time = []
step_time = []
start_time = time.time()
steps_per_epoch = (len(int_words) * skip_window) // batch_size
current_batch = 0
vec_params = []
folder = "results/{0}_{1}".format(skip_window, init_lr)
try:
	os.mkdir(folder)
except:
	pass
open(folder + "/params.txt", mode='w')
for i in range(steps_per_epoch * n_epoch):
	sgd.zero_grad()
	x1, x2, y = next(gen)
	out = net.forward(x2, x1, y)
	out.backward() 
	sgd.step()
	n_words = i * batch_size
	lr = max(.0001, init_lr * (1.0 - n_words /
							   (len(int_words) * skip_window * n_epoch)))
	for param_group in sgd.param_groups:
		param_group['lr'] = lr
	losses.append(out.detach().cpu().numpy())
	if i % (steps_per_epoch // 3) == 0 and i > 0:
		print(net.alpha, net.beta)
		s = "Loss: {0:.4f} lr: {1:.4f} Time Left: {2:.2f}"
		span = (time.time() - start_time)
		print(s.format(np.mean(losses), lr, span))
		start_time = time.time()

		vocab = list(word2int.keys())
		embed_dict = {}
		embed_dict_2 = {}
		embed_dict_3 = {}
		alpha, beta = net.alpha.detach().numpy()[0], net.beta.detach().numpy()[0]
		for i in range(len(vocab)):
			word = vocab[i]
			con_mat, vow_mat = word2vec_seperated(
				char2tup, word, n_chars, n_consonant, n_vowel)
			word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
				(1, 1, n_chars, (n_consonant + n_vowel)))
			x_index = word2int[word]
			em_row1, em_row2 = net.get_embedding(word_mat, [x_index])
			embed_dict[word] = em_row1.reshape((-1,))
			embed_dict_2[word] = em_row2.reshape((-1,))
			embed_dict_3[word] = alpha * embed_dict[word] + beta * embed_dict_2[word]

		
		vec_params.append((alpha, beta))
		net.save_embedding(embed_dict, folder + "/w2v_cnn1_{0}.txt".format(current_batch), device)
		net.save_embedding(embed_dict_2, folder + "/w2v_cnn2_{0}.txt".format(current_batch), device)
		net.save_embedding(embed_dict_3, folder + "/w2v_cnn3_{0}.txt".format(current_batch), device)
		current_batch += 1
		open(folder + "/params.txt", mode='a').write('{0} {1}\n'.format(alpha, beta))
		
