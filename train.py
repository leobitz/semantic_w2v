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
from models.preSG import PRESG
from models.sg import SG
from models.CNN import CNN
from models.CNN_SG_out import CNNSG_OUT
import os


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", action="store", type=int)
parser.add_argument("-w", "--window", action="store", type=int)
parser.add_argument("-s", "--size", action="store", type=int)
parser.add_argument("-m", "--model", action="store")
parser.add_argument("-f", "--folder", action="store")
parser.add_argument("-d", "--data", action="store")
parser.add_argument("-l", "--log", action="store", type=int)
parser.add_argument("-v", "--save", action="store", type=int)
parser.add_argument("-p", "--pre", action="store", default='')
args = parser.parse_args()


def save_result(step):
    vocab = list(word2int.keys())
    result_dicts = []
    for i in range(len(vocab)):
        word = vocab[i]
        # con_mat, vow_mat = word2vec_seperated(
        #     char2tup, word, n_chars, n_consonant, n_vowel
        # )
        # word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
        #     (1, 1, n_chars, (n_consonant + n_vowel))
        # )
        x_index = word2int[word]
        # word_image = np.random.rand(embed_size)
        result = net.get_embedding(x_index)
        for j in range(len(result)):
            if len(result_dicts) < j + 1:
                result_dicts.append({})
            result_dicts[j][word] = result[j].reshape((-1,))

    for counter, rdict in enumerate(result_dicts):
        net.save_embedding(rdict, "{0}_{1}".format(step, counter))


def generateSG(data, win_size, batch_size):
    state = win_size
    while True:
        context = data[state - win_size : state + win_size + 1]
        target = context.pop(win_size)
        targets = [target] * (win_size)
        contexts = random.sample(context, win_size)
        state += 1
        if state >= len(data) - win_size:
            state = win_size
        yield targets, contexts


def getPre2Tensor(file, delimiter=" "):
    lines = open(file, encoding="utf8").readlines()
    vocab_size, embed_size = [int(s) for s in lines[0].split()]
    embeddings = {}
    for i in range(1, vocab_size):
        try:
            line = lines[i][:-1].split(delimiter)
            word = line[0]
            if word in word2int:
                wordvec = np.array([np.float64(j) for j in line[1:] if j != ""])
                embeddings[word2int[word]] = t.tensor(wordvec)
        except Exception as e:
            print(lines[i])
            print(e)
    return embeddings


def getWord2Tensor(vocab, char2tup, n_chars, n_consonant, n_vowel):
    images = {}
    for word in vocab:
        con_mat, vow_mat = word2vec_seperated(
            char2tup, word, n_chars, n_consonant, n_vowel
        )
        word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
            (1, 1, n_chars, (n_consonant + n_vowel))
        )
        images[word2int[word]] = t.tensor(word_mat).double()
    return images


def get_pretrained_image(x):
    return word_images[x]


def get_cnn_image(x):
    return word_images[x]


n_chars = 11 + 2
n_epoch = args.epochs
skip_window = args.window
batch_size = 1
embed_size = args.size
init_lr = 0.025
save_folder = args.folder
save_name = "results/" + save_folder + "/"
data_file = args.data

words = read_file(data_file)
words, word2freq = min_count_threshold(words)
# words = subsampling(words, 1e-3)
vocab, word2int, int2word = build_vocab(words)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
ns_unigrams = np.array(ns_sample(word2freq, word2int, int2word, 0.75), dtype=np.int32)
print("Words to train: ", len(words))
print("Vocabs to train: ", len(vocab))

int_words = words_to_ints(word2int, words)
int_words = np.array(int_words, dtype=np.int32)

word_images = None

if args.pre != '':
    word_images = getPre2Tensor(args.pre)
if "cnn" in args.model:
    word_images = getWord2Tensor(vocab, char2tup, n_chars, n_consonant, n_vowel)

if not os.path.exists(save_name):
    os.mkdir(save_name)
get_image = get_cnn_image
if args.model == "sg":
    model = SG
    get_image = None
elif args.model == "pre":
    model = PRESG
    get_image = get_pretrained_image
elif args.model == "cnnsg":
    model = CNNSG
    get_image = get_cnn_image
elif args.model == "cnn":
    model = CNN
    get_image = get_cnn_image
elif args.model == "cnn_sg_out":
    model = CNNSG_OUT
    get_image = get_cnn_image
else:
    model = SG
    get_image = None


gen = generateSG(list(int_words), skip_window, batch_size)
net = model(
    neg_dist=ns_unigrams,
    embed_size=embed_size,
    vocab_size=len(vocab),
    get_image=get_image,
)
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
save_per_epoch = args.save
logs_per_epoch = args.log
total_time = 0


word_image = np.random.rand(embed_size)
while trained_steps <= total_steps:

    batch_x, batch_y = next(gen)
    for j in range(skip_window):
        x, y = batch_x[j], batch_y[j]
        sgd.zero_grad()
        out = net.forward(x, y)
        out.backward()
        sgd.step()

    lr = max(0.0001, init_lr * (1.0 - trained_steps / total_steps))
    for param_group in sgd.param_groups:
        param_group["lr"] = lr

    losses.append(out.detach().numpy())
    if trained_steps != 0 and trained_steps % (total_words // save_per_epoch) == 0:
        epoch = trained_steps // total_words
        file_name = "{0}_{1}_{2}".format(epoch, skip_window, embed_size)
        save_result(save_name + file_name)
        print("Saving weights. ", (trained_steps / total_words))

    if trained_steps != 0 and (trained_steps % (total_words // logs_per_epoch)) == 0:
        span = time.time() - start_time
        rate = span / (total_words // logs_per_epoch)
        left = rate * ((total_steps - trained_steps) // logs_per_epoch)
        s = "Progress: {3:.2f}% Loss {0:.4f} lr: {1:.4f} Time Left: {2:.2f}s"
        print(s.format(np.mean(losses), lr, left, (trained_steps * 100 / total_steps)))
        losses.clear()
        start_time = time.time()

    trained_steps += 1
