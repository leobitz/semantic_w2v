import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from data_handle import *
from torch.autograd import Variable
import argparse
from models.CNN_SG import CNNSG
from models.sg import SG
import os


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", action="store", type=int)
parser.add_argument("-w", "--window", action="store", type=int)
parser.add_argument("-s", "--size", action="store", type=int)
parser.add_argument("-m", "--model", action="store")
parser.add_argument("-f", "--folder", action="store")
args = parser.parse_args()


def save_result(step):
    vocab = list(word2int.keys())
    result_dicts = []
    for i in range(len(vocab)):
        word = vocab[i]
        con_mat, vow_mat = word2vec_seperated(
            char2tup, word, n_chars, n_consonant, n_vowel
        )
        word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
            (1, 1, n_chars, (n_consonant + n_vowel))
        )
        x_index = word2int[word]
        result = net.get_embedding(word_mat, [x_index])
        for j in range(len(result)):
            if len(result_dicts) < j + 1:
                result_dicts.append({})
            result_dicts[j][word] = result[j].reshape((-1,))

    for counter, rdict in enumerate(result_dicts):
        net.save_embedding(
            rdict, "results/w2v_high_{0}_{1}.txt".format(counter, step), device
        )


def generateSG(data, win_size, batch_size):
    state = win_size
    while True:
        context = data[state - win_size : state + win_size + 1]
        target = context.pop(win_size)
        targets = [target] * (win_size)
        contexts = random.sample(context, win_size)

        if state >= len(data) - win_size:
            state = win_size
        yield targets, contexts


print(args.epochs, args.size, args.window)

words = read_file("data/news.txt") [:10_000]
# open('data/news_mini.txt', mode='w', encoding='utf-8').write(" ".join(words))
words, word2freq = min_count_threshold(words)
# words = subsampling(words, 1e-3)
vocab, word2int, int2word = build_vocab(words)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
ns_unigrams = np.array(ns_sample(word2freq, word2int, int2word, 0.75), dtype=np.int32)
print("Words to train: ", len(words))
print("Vocabs to train: ", len(vocab))


int_words = words_to_ints(word2int, words)
int_words = np.array(int_words, dtype=np.int32)
n_chars = 11 + 2
n_epoch = args.epochs
skip_window = args.window
batch_size = 1
embed_size = args.size
init_lr = 0.025
save_folder = args.folder
save_name = "results/" + save_folder + "/"
if not os.path.exists(save_name):
    os.mkdir(save_name)


if args.model == "sg":
    model = SG
elif args.model == "cnnsg":
    model = CNNSG
else:
    model = SG

gen = generateSG(list(int_words), skip_window, batch_size)

net = model(neg_dist=ns_unigrams, embed_size=embed_size, vocab_size=len(vocab))

sgd = optimizers.SGD(net.parameters(), lr=init_lr)
start = time.time()
losses = []
grad_time = []
forward_time = []
backward_time = []
step_time = []
start_time = time.time()
total_words = len(int_words)
total_steps = total_words * n_epoch
trained_steps = 0

steps_per_epoch = total_words * skip_window
word_image = np.random.rand(embed_size)
while trained_steps < total_steps:

    batch_x, batch_y = next(gen)
    for j in range(skip_window):
        x, y = batch_x[j], batch_y[j]
        sgd.zero_grad()
        out = net.forward(x, y, word_image)
        out.backward()
        sgd.step()

    lr = max(0.0001, init_lr * (1.0 - trained_steps / total_steps))
    for param_group in sgd.param_groups:
        param_group["lr"] = lr

    losses.append(out.detach().numpy())
    if trained_steps % (total_steps // n_epoch) == 0:
        span = time.time() - start_time
        s = "Progress: {3:.2f}% Loss {0:.4f} lr: {1:.4f} Time Left: {2:.2f}"
        print(s.format(np.mean(losses), lr, span, (trained_steps * 100 / total_steps)))
        start_time = time.time()

    if (trained_steps > 0) and (trained_steps % (total_steps // n_epoch) == 0):
        print("Loss: ", np.mean(losses))
        losses = []
        save_result(save_name + (i // steps_per_epoch))

    trained_steps += 1

save_result(save_name + n_epoch)
