import collections
from collections import Counter
import numpy as np
import random
from gensim_wrapper import *


def read_file(filename='data/news.txt'):
    """
    returns list of tokens in the corpus

    """
    return open(filename, encoding='utf8').read().strip().split(' ')


def words_to_ints(word2int, words):
    """
    returns integer representation of the tokes by mapping using word2int

    """
    return [word2int[word] for word in words]


def get_frequency(words, word2int, int2word):
    """
    returns dictionary of words with thier frequencies

    """
    word2freq = {}
    for word in words:
        if word not in word2freq:
            word2freq[word] = 0
        word2freq[word] += 1.0
    return word2freq


def min_count_threshold(words, word2freq, min_count=5):
    new_words = []
    for word in words:
        if word2freq[word] >= min_count:
            new_words.append(word)
        else:
            new_words.append("###")
    assert len(new_words) == len(words)
    return new_words


def ns_sample(word2freq, word2int, int2word, rate):
    """
    returns the negative sampling distribution

    """
    unigrams = np.array(list(word2freq.values()), dtype=np.float32)
    # unigrams = np.power(unigrams, rate)
    table = []
    p = 0
    i = 0
    total_p = unigrams.sum()
    for word in word2freq.keys():
        count = word2freq[word]
        p += float(pow(count, rate) / total_p)
        while (float(i) / total_p) < p:
            table.append(word2int[word])
            i += 1
    return table


def min_count_threshold(words, min_count=5):
    new_words = []
    word2freq = {}
    unkown_word = "*###*"
    new_words.append(unkown_word)
    for word in words:
        if word not in word2freq:
            word2freq[word] = 0
        word2freq[word] += 1

    freq = {}
    freq[unkown_word] = 0
    for word in words:
        if word2freq[word] >= min_count:
            new_words.append(word)
            if word not in freq:
                freq[word] = 0
            freq[word] += 1
        else:
            freq[unkown_word] += 1

    return new_words, freq


def build_vocab(words):
    """
    returns vocabulary, word mapped to integer and integer-word mapping

    """
    word2int = {}
    int2word = {}
    for word in words:
        if word not in word2int:
            index = len(word2int)
            int2word[index] = word
            word2int[word] = index
    vocab = list(word2int.keys())
    return vocab, word2int, int2word


def build_charset():
    """
    returns charcter mapping to integer and vice versa

    """
    charset = open('data/charset.txt', encoding='utf-8').readlines()
    n_consonant = len(charset)
    n_vowel = 10
    char2int, int2char, char2tup, tup2char = {}, {}, {}, {}
    j = 0
    for k in range(len(charset)):
        row = charset[k][:-1].split(',')
        for i in range(len(row)):
            char2tup[row[i]] = (k, i)
            int2char[j] = row[i]
            char2int[row[i]] = j
            tup = "{0}-{1}".format(k, i)
            tup2char[tup] = row[i]
            j += 1
    return char2int, int2char, char2tup, tup2char, n_consonant, n_vowel


def word2vec_seperated(char2tup, word, max_word_len, n_consonant, n_vowel):
    """
    using char2int mapping, it creates stack of one-hot vectors characters for a word
    the mapping is seperated for vowel and consonant
    [0 0 0 1 0 0: 0 1 0 0]
    Consonan        Vowel
    """
    cons = np.zeros((max_word_len, n_consonant), dtype=np.float32)
    vowel = np.zeros((max_word_len, n_vowel), dtype=np.float32)
    for i in range(len(word)):
        char = word[i]
        t = char2tup[char]
        cons[i][t[0]] = 1
        vowel[i][t[1]] = 1
    con, vow = char2tup[' ']
    cons[i + 1:, con] = 1
    vowel[i + 1:, vow] = 1
    return cons, vowel

def char_to_vec(char, char2tup, n_consonant, n_vowel):
    vec = np.zeros(((n_consonant + n_vowel), ))
    t = char2tup[char]
    vec[t[0]] = 1
    vec[n_consonant + t[1]] = 1
    return vec


def word2vec_single(char2tup, words, max_word_len, n_consonant, n_vowel):
    """
    using char2int mapping, it creates stack of one-hot vectors characters for a word
    the mapping is seperated for vowel and consonant
    [0 0 0 1 0 0: 0 1 0 0]
    Consonan        Vowel
    """
    inputs = [np.ndarray((len(words), (n_consonant + n_vowel))) for i in range(max_word_len)]
    for i in range(len(words)):
        chars = words[i] 
        for j in range(len(chars)):
            inputs[j][i] = char_to_vec(chars[j], char2tup, n_consonant, n_vowel)
        for k in range(j + 1, max_word_len):
            inputs[k][i] = char_to_vec(chars[j], char2tup, n_consonant, n_vowel)
    return inputs



def char_to_vec(char, char2tup, n_consonant, n_vowel):
    vec = np.zeros(((n_consonant + n_vowel), ))
    t = char2tup[char]
    vec[t[0]] = 1
    vec[n_consonant + t[1]] = 1
    return vec


def word2vec_single(char2tup, words, max_word_len, n_consonant, n_vowel):
    """
    using char2int mapping, it creates stack of one-hot vectors characters for a word
    the mapping is seperated for vowel and consonant
    [0 0 0 1 0 0: 0 1 0 0]
    Consonan        Vowel
    """
    inputs = [np.ndarray((len(words), (n_consonant + n_vowel)))
              for i in range(max_word_len)]
    for i in range(len(words)):
        chars = words[i]
        for j in range(len(chars)):
            inputs[j][i] = char_to_vec(
                chars[j], char2tup, n_consonant, n_vowel)
        for k in range(j + 1, max_word_len):
            inputs[k][i] = char_to_vec(
                chars[j], char2tup, n_consonant, n_vowel)
    return inputs


def word2vec(char2int, word, max_word_len):
    """
    using char2int mapping, it creates stack of one-hot vectors characters for a word
    [0 0 0 1 0 0 0 0 0 0]
    """
    max_n_char = len(char2int)
    vec = np.zeros((max_word_len, max_n_char), dtype=np.float32)
    for i in range(len(word)):
        char = word[i]
        t = char2int[char]
        vec[i][t] = 1
    spacei = char2int[' ']
    vec[i + 1:, spacei] = 1
    return vec


def one_hot(n, size):
    v = np.zeros((size,))
    v[n] = 1
    return v


def one_hot_decode(int2word, vec):
    indexes = np.argmax(vec, axis=1)
    words = []
    for i in indexes:
        words.append(int2word[i])
    return words


def sentense_to_vec(words):
    """
    creates stack of word matrix created by word2vec() function for sentense

    """
    vecs = []
    for w in words:
        vecs.append(word2vec(w))
    vec = np.concatenate(vecs)
    return vec


def generate_batch_embed(data, batch_size, skip_window):
    """

    returns cbow input of integer sequence of words

    """
    assert batch_size % skip_window == 0
    ci = skip_window  # current_index
    while True:
        batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        batch_index = 0
        for batch_index in range(0, batch_size, skip_window):  # fill the batch inputs
            context = data[ci - skip_window:ci + skip_window + 1]
            # remove the target from context words
            target = context.pop(skip_window)
            # context = random.sample(context, skip_window * 2)
            context = np.random.choice(context, skip_window, replace=False)
            batch_inputs[batch_index:batch_index +
                         skip_window] = context
            batch_labels[batch_index:batch_index + skip_window, 0] = target
            ci += 1
        if len(data) - ci - skip_window < batch_size:
            ci = skip_window
        yield batch_inputs, batch_labels


def generate_batch_embed_v2(data, embeddings, batch_size, skip_window):
    """

    returns cbow input of integer sequence of words and the seq_embeddings of the context words

    """
    assert batch_size % skip_window == 0
    ci = skip_window  # current_index
    # embeddings = normalize(embeddings)
    while True:
        batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        batch_embeddings = np.ndarray(
            shape=(batch_size, embeddings.shape[1]), dtype=np.float32)
        batch_index = 0
        win = skip_window * 2
        for batch_index in range(0, batch_size, win):  # fill the batch inputs
            context = data[ci - skip_window:ci + skip_window + 1]
            # remove the target from context words
            target = context.pop(skip_window)
            # context = random.sample(context, skip_window * 2)
            # context = np.random.choice(context, skip_window, replace=False)
            batch_embeddings[batch_index:batch_index +
                             win] = embeddings[target]
            batch_inputs[batch_index:batch_index + win] = target
            batch_labels[batch_index:batch_index + win, 0] = context
            ci += 1
        if len(data) - ci - skip_window < batch_size:
            ci = skip_window
        yield batch_inputs, batch_labels, batch_embeddings


def generate_batch_input_dense(data, embeddings, batch_size, skip_window, embed_size=28):
    """

    returns cbow input of integer sequence of words and the seq_embeddings of the context words

    """
    assert batch_size % skip_window == 0
    ci = 0
    embeddings = normalize(embeddings)
    while True:
        batch_contexts = np.ndarray(
            shape=(batch_size, embed_size), dtype=np.float32)
        batch_targets = np.ndarray(
            shape=(batch_size, embed_size), dtype=np.float32)
        batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for batch_index in range(0, batch_size, skip_window):  # fill the batch inputs
            context, ci = get_context_words(data, ci, skip_window * 2 + 1)
            ci += 1
            target = context.pop(skip_window)
            context = np.random.choice(context, skip_window, replace=False)
            batch_contexts[batch_index:batch_index +
                           skip_window] = embeddings[context]
            batch_targets[batch_index:batch_index +
                          skip_window] = embeddings[target]
            batch_labels[batch_index:batch_index + skip_window, 0] = target

        yield batch_contexts, batch_targets, batch_labels


def normalize(array):
    if len(array.shape) == 1:
        return array / np.linalg.norm(array)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / norms


# def generate_batch_embed_v3(data, embeddings, batch_size, skip_window):
#      """

#     returns cbow input of integer sequence of words and the seq_embeddings of the context words

#     """
#     assert batch_size % skip_window == 0
#     ci = skip_window  # current_index
#     while True:
#         batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
#         batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#         batch_embeddings = np.ndarray(shape=(batch_size, embeddings.shape[1]), dtype=np.int32)
#         batch_index = 0
#         for batch_index in range(0, batch_size, skip_window):  # fill the batch inputs
#             context = data[ci - skip_window:ci + skip_window + 1]
#             # remove the target from context words
#             target = context.pop(skip_window)
#             # context = random.sample(context, skip_window * 2)
#             context = np.random.choice(context, skip_window, replace=False)
#             batch_embeddings[batch_index:batch_index +
#                              skip_window] = embeddings[context]
#             batch_inputs[batch_index:batch_index +
#                          skip_window] = context
#             batch_labels[batch_index:batch_index + skip_window, 0] = target
#             ci += 1
#         if len(data) - ci - skip_window < batch_size:
#             ci = skip_window
#         yield batch_inputs, batch_labels, batch_embeddings


def get_context_words(words, start, length):
    """

    returns slice of words from the full corpus from start to start + length

    """
    if start + length > len(words):
        start = 0
    end = start + length
    return words[start:end], start


def generate_batch_image_v2(words, word2int, char2int, batch_size, skip_window):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    assert batch_size % (skip_window * 2) == 0
    ci = 0
    window = skip_window * 2 + 1
    targets, target_inputs = {}, {}
    for word in word2int.keys():
        target = word + '&'
        target_input = '&' + target
        targets[word] = target
        target_inputs[word] = target_input
    while True:
        batch_inputs = np.ndarray(
            shape=(batch_size, 13, 309), dtype=np.float32)
        batch_outputs = np.ndarray(
            shape=(batch_size, 13, 309), dtype=np.float32)
        batch_raw_inputs = np.ndarray(
            shape=(batch_size, 13, 309, 1), dtype=np.float32)
        batch_index = 0
        for batch_index in range(0, batch_size, skip_window * 2):  # fill the batch inputs
            context, ci = get_context_words(words, ci, window)
            target_word = context.pop(skip_window)
            target_vec = word2vec(char2int, targets[target_word], 13)
            target_input_vec = word2vec(
                char2int, target_inputs[target_word], 13)  # .reshape((13, 309, 1))
            for i in range(len(context)):
                batch_inputs[i + batch_index] = target_input_vec
                batch_outputs[i + batch_index] = target_vec
                batch_raw_inputs[i +
                                 batch_index] = word2vec(char2int, context[i], 13).reshape((13, 309, 1))

        yield [batch_raw_inputs, batch_inputs], batch_outputs


def generate_batch_image_v3(words, word2int, char2int, batch_size, skip_window):
    """

    returns skip-gram input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    assert batch_size % (skip_window * 2) == 0
    ci = 0
    window = skip_window * 2 + 1
    targets, target_inputs = {}, {}
    for word in word2int.keys():
        target = word + '&'
        target_input = '&' + target
        targets[word] = target
        target_inputs[word] = target_input
    while True:
        batch_inputs = np.ndarray(
            shape=(batch_size, 13, 309), dtype=np.float32)
        batch_outputs = np.ndarray(
            shape=(batch_size, 13, 309), dtype=np.float32)
        batch_raw_inputs = np.ndarray(
            shape=(batch_size, 13, 309, 1), dtype=np.float32)
        batch_index = 0
        for batch_index in range(0, batch_size, skip_window * 2):  # fill the batch inputs
            context, ci = get_context_words(words, ci, window)
            target_word = context.pop(skip_window)
            target_vec = word2vec(char2int, target_word,
                                  13).reshape((13, 309, 1))
            for i in range(len(context)):
                batch_inputs[i + batch_index] = word2vec(
                    char2int, target_inputs[context[i]], 13)
                batch_outputs[i +
                              batch_index] = word2vec(char2int, targets[context[i]], 13)
                batch_raw_inputs[i + batch_index] = target_vec

        yield [batch_raw_inputs, batch_inputs], batch_outputs


def generate_word_images_feat(words, word2nt, char2int, embeddings, batch_size):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output with features

    """
    targets, target_inputs = [], []
    for word in words:
        target = word + '|'
        target_input = '&' + target
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]
        batch_inputs = []
        batch_outputs = []
        batch_raw_inputs = []
        batch_embeddings = np.ndarray((batch_size, embeddings.shape[1]))
        for i in range(batch_size):
            target = word2vec(char2int, batch_targets[i], 13)
            target_in = word2vec(char2int, batch_target_ins[i], 13)
            word = word2vec(char2int, batch_words[i],  13)
            batch_inputs.append(target)
            batch_outputs.append(target_in)
            batch_raw_inputs.append(word)
            batch_embeddings[i] = embeddings[word2nt[batch_words[i]]]
        # .reshape((batch_size, 13, 309, 1))
        batch_inputs = np.stack(batch_inputs)
        # .reshape((batch_size, 13, 309, 1))
        batch_outputs = np.stack(batch_outputs)
        batch_raw_inputs = np.stack(batch_raw_inputs).reshape(
            (batch_size, 13, 309, 1))
        yield [batch_raw_inputs, batch_inputs, batch_embeddings], batch_outputs
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_word_images(words, char2int, batch_size):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word + '|'
        target_input = '&' + target
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]
        batch_inputs = []
        batch_outputs = []
        batch_raw_inputs = []
        for i in range(batch_size):
            target = word2vec(char2int, batch_targets[i], 13)
            target_in = word2vec(char2int, batch_target_ins[i], 13)
            word = word2vec(char2int, batch_words[i],  13)
            batch_inputs.append(target)
            batch_outputs.append(target_in)
            batch_raw_inputs.append(word)
        # .reshape((batch_size, 13, 309, 1))
        batch_inputs = np.stack(batch_inputs)
        # .reshape((batch_size, 13, 309, 1))
        batch_outputs = np.stack(batch_outputs)
        batch_raw_inputs = np.stack(batch_raw_inputs).reshape(
            (batch_size, 13, 309, 1))
        yield [batch_raw_inputs, batch_inputs], batch_outputs
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_word_images_flat(words, char2tup, batch_size, max_char_len, n_consonant, n_vowels=10):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    ci = 0
    while True:
        batch_input = np.ndarray((batch_size, 13 * (n_consonant + n_vowels)))
        words, ci = get_context_words(words, ci, batch_size)
        for i in range(batch_size):
            con_vec, vow_vec = word2vec_seperated(
                char2tup, words[i], max_char_len, n_consonant, n_vowels)
            batch_input[i] = np.concatenate(
                [con_vec, vow_vec], axis=1).flatten()
        ci += batch_size
        yield batch_input, batch_input


def generate_word_images_multi(words, char2tup, batch_size, n_consonant, n_vowels):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word  # + '|'
        target_input = '&' + target + '|'
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    n_chars = 13
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]

        batch_inputs = np.ndarray(
            (batch_size, n_chars, (n_consonant + n_vowels), 1), dtype=np.float32)
        batch_cons_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_vowels), dtype=np.float32)
        batch_cons_output = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_output = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        for i in range(batch_size):
            input_con, input_vow = word2vec_seperated(char2tup,
                                                      batch_words[i], n_chars, n_consonant, n_vowels)
            target_con, target_vow = word2vec_seperated(char2tup,
                                                        batch_targets[i], n_chars, n_consonant, n_vowels)
            decoder_con, decoder_vow = word2vec_seperated(char2tup,
                                                          batch_target_ins[i], n_chars, n_consonant, n_vowels)

            batch_inputs[i] = np.concatenate([input_con, input_vow], axis=1).reshape(
                (n_chars, (n_consonant + n_vowels), 1))
            batch_cons_dec_inputs[i] = decoder_con
            batch_vow_dec_inputs[i] = decoder_vow
            batch_cons_output[i] = target_con
            batch_vow_output[i] = target_vow
        yield [batch_inputs, batch_cons_dec_inputs, batch_vow_dec_inputs], [batch_cons_output, batch_vow_output]
        batch += 1
        if batch == n_batchs:
            batch = 0

def generate_word_images_multi_v6(words, char2tup, batch_size, n_consonant, n_vowels):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word  # + '|'
        target_input = '&' + target + '|'
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    n_chars = 13
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]
        batch_words_string = []
        batch_inputs = np.ndarray(
            (batch_size, n_chars, (n_consonant + n_vowels), 1), dtype=np.float32)
        batch_cons_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_vowels), dtype=np.float32)
        batch_cons_output = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_output = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        for i in range(batch_size):
            batch_words_string.append(batch_words[i])
            # input_con, input_vow = word2vec_seperated(char2tup,
            #                                           batch_words[i], n_chars, n_consonant, n_vowels)
            target_con, target_vow = word2vec_seperated(char2tup,
                                                        batch_targets[i], n_chars, n_consonant, n_vowels)
            decoder_con, decoder_vow = word2vec_seperated(char2tup,
                                                          batch_target_ins[i], n_chars, n_consonant, n_vowels)

            # batch_inputs[i] = np.concatenate([input_con, input_vow], axis=1).reshape(
            #     (n_chars, (n_consonant + n_vowels), 1))
            batch_cons_dec_inputs[i] = decoder_con
            batch_vow_dec_inputs[i] = decoder_vow
            batch_cons_output[i] = target_con
            batch_vow_output[i] = target_vow
        batch_inputs = word2vec_single(char2tup, batch_words_string, 13, n_consonant, n_vowels)
        batch_words_string.clear()
        yield batch_inputs + [batch_cons_dec_inputs, batch_vow_dec_inputs], [batch_cons_output, batch_vow_output]
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_word_images_multi_v5(words, char2tup, batch_size, n_consonant, n_vowels):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word  # + '|'
        target_input = '&' + target + '|'
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    n_chars = 13
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]

        batch_inputs = np.ndarray(
            (batch_size, n_chars, (n_consonant + n_vowels), 1), dtype=np.float32)
        batch_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant + n_vowels), dtype=np.float32)
        batch_cons_output = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_output = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        for i in range(batch_size):
            input_con, input_vow = word2vec_seperated(char2tup,
                                                      batch_words[i], n_chars, n_consonant, n_vowels)
            target_con, target_vow = word2vec_seperated(char2tup,
                                                        batch_targets[i], n_chars, n_consonant, n_vowels)
            decoder_con, decoder_vow = word2vec_seperated(char2tup,
                                                          batch_target_ins[i], n_chars, n_consonant, n_vowels)

            batch_inputs[i] = np.concatenate([input_con, input_vow], axis=1).reshape(
                (n_chars, (n_consonant + n_vowels), 1))
            batch_dec_inputs[i] = np.concatenate([decoder_con, decoder_vow], axis=1).reshape(
                (n_chars, (n_consonant + n_vowels)))
            batch_cons_output[i] = target_con
            batch_vow_output[i] = target_vow
        yield [batch_inputs, batch_dec_inputs], [batch_cons_output, batch_vow_output]
        batch += 1
        if batch == n_batchs:
            batch = 0

def generate_word_images_multi_v6(words, char2tup, batch_size, n_consonant, n_vowels):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word  # + '|'
        target_input = '&' + target + '|'
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    n_chars = 13
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]
        batch_words_string = []
        batch_inputs = np.ndarray(
            (batch_size, n_chars, (n_consonant + n_vowels), 1), dtype=np.float32)
        batch_cons_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_vowels), dtype=np.float32)
        batch_cons_output = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_output = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        for i in range(batch_size):
            batch_words_string.append(batch_words[i])
            # input_con, input_vow = word2vec_seperated(char2tup,
            #                                           batch_words[i], n_chars, n_consonant, n_vowels)
            target_con, target_vow = word2vec_seperated(char2tup,
                                                        batch_targets[i], n_chars, n_consonant, n_vowels)
            decoder_con, decoder_vow = word2vec_seperated(char2tup,
                                                          batch_target_ins[i], n_chars, n_consonant, n_vowels)

            # batch_inputs[i] = np.concatenate([input_con, input_vow], axis=1).reshape(
            #     (n_chars, (n_consonant + n_vowels), 1))
            batch_cons_dec_inputs[i] = decoder_con
            batch_vow_dec_inputs[i] = decoder_vow
            batch_cons_output[i] = target_con
            batch_vow_output[i] = target_vow
        batch_inputs = word2vec_single(
            char2tup, batch_words_string, 13, n_consonant, n_vowels)
        batch_words_string.clear()
        yield batch_inputs + [batch_cons_dec_inputs, batch_vow_dec_inputs], [batch_cons_output, batch_vow_output]
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_word_images_multi_v5(words, char2tup, batch_size, n_consonant, n_vowels):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word  # + '|'
        target_input = '&' + target + '|'
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    n_chars = 13
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]

        batch_inputs = np.ndarray(
            (batch_size, n_chars, (n_consonant + n_vowels), 1), dtype=np.float32)
        batch_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant + n_vowels), dtype=np.float32)
        batch_cons_output = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_output = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        for i in range(batch_size):
            input_con, input_vow = word2vec_seperated(char2tup,
                                                      batch_words[i], n_chars, n_consonant, n_vowels)
            target_con, target_vow = word2vec_seperated(char2tup,
                                                        batch_targets[i], n_chars, n_consonant, n_vowels)
            decoder_con, decoder_vow = word2vec_seperated(char2tup,
                                                          batch_target_ins[i], n_chars, n_consonant, n_vowels)

            batch_inputs[i] = np.concatenate([input_con, input_vow], axis=1).reshape(
                (n_chars, (n_consonant + n_vowels), 1))
            batch_dec_inputs[i] = np.concatenate([decoder_con, decoder_vow], axis=1).reshape(
                (n_chars, (n_consonant + n_vowels)))
            batch_cons_output[i] = target_con
            batch_vow_output[i] = target_vow
        yield [batch_inputs, batch_dec_inputs], [batch_cons_output, batch_vow_output]
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_word_images_multi_v4(words, char2tup, batch_size, n_consonant, n_vowels):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word  # + '|'
        target_input = '&' + target + '|'
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    n_chars = 13
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]

        batch_inputs_cons = np.ndarray(
            (batch_size, n_chars, n_consonant, 1), dtype=np.float32)
        batch_inputs_vow = np.ndarray(
            (batch_size, n_chars, n_vowels, 1), dtype=np.float32)
        batch_cons_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_vowels), dtype=np.float32)
        batch_cons_output = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_output = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        for i in range(batch_size):
            input_con, input_vow = word2vec_seperated(char2tup,
                                                      batch_words[i], n_chars, n_consonant, n_vowels)
            target_con, target_vow = word2vec_seperated(char2tup,
                                                        batch_targets[i], n_chars, n_consonant, n_vowels)
            decoder_con, decoder_vow = word2vec_seperated(char2tup,
                                                          batch_target_ins[i], n_chars, n_consonant, n_vowels)

            batch_inputs_cons[i] = input_con.reshape(
                (-1, n_chars, n_consonant, 1))
            batch_inputs_vow[i] = input_vow.reshape((-1, n_chars, n_vowels, 1))
            batch_cons_dec_inputs[i] = decoder_con
            batch_vow_dec_inputs[i] = decoder_vow
            batch_cons_output[i] = target_con
            batch_vow_output[i] = target_vow
        yield [batch_inputs_cons, batch_inputs_vow, batch_cons_dec_inputs, batch_vow_dec_inputs], [batch_cons_output, batch_vow_output]
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_word_images_multi_v2(words, char2int, char2tup, batch_size, n_consonant, n_vowels):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word  # + '|'
        target_input = '&' + target + '|'
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    n_chars = 13
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]

        batch_inputs = np.ndarray(
            (batch_size, n_chars, (n_consonant + n_vowels), 1), dtype=np.float32)
        batch_cons_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_vowels), dtype=np.float32)
        batch_output = np.ndarray(
            (batch_size, n_chars, len(char2tup)), dtype=np.float32)
        for i in range(batch_size):
            input_con, input_vow = word2vec_seperated(char2tup,
                                                      batch_words[i], n_chars, n_consonant, n_vowels)
            target_output = word2vec(char2int, batch_targets[i], 13)
            decoder_con, decoder_vow = word2vec_seperated(char2tup,
                                                          batch_target_ins[i], n_chars, n_consonant, n_vowels)

            batch_inputs[i] = np.concatenate([input_con, input_vow], axis=1).reshape(
                (n_chars, (n_consonant + n_vowels), 1))
            batch_cons_dec_inputs[i] = decoder_con
            batch_vow_dec_inputs[i] = decoder_vow
            batch_output[i] = target_output
        yield [batch_inputs, batch_cons_dec_inputs, batch_vow_dec_inputs], batch_output
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_word_images_rnn(words, char2int, batch_size):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    targets, target_inputs = [], []
    for word in words:
        target = word + '&'
        target_input = '&' + target
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]
        batch_inputs = []
        batch_outputs = []
        batch_raw_inputs = []
        for i in range(batch_size):
            target = word2vec(char2int, batch_targets[i], 13)
            target_in = word2vec(char2int, batch_target_ins[i], 13)
            word = word2vec(char2int, batch_words[i],  13)
            batch_inputs.append(target)
            batch_outputs.append(target_in)
            batch_raw_inputs.append(word)
        # .reshape((batch_size, 13, 309, 1))
        batch_inputs = np.stack(batch_inputs)
        # .reshape((batch_size, 13, 309, 1))
        batch_outputs = np.stack(batch_outputs)
        batch_raw_inputs = np.stack(batch_raw_inputs)
        yield [batch_raw_inputs, batch_inputs], batch_outputs
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_batch_rnn_v2(data, int2word, char2int, batch_size, skip_window, n_chars, n_features):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    assert batch_size % (skip_window * 2) == 0
    ci = skip_window  # current_index
    while True:
        batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(batch_size), dtype=np.int32)
        rnn_inputs = np.ndarray(
            shape=(batch_size, n_chars, n_features), dtype=np.float32)
        rnn_outputs = np.ndarray(
            shape=(batch_size, n_chars, n_features), dtype=np.float32)
        batch_index = 0
        for batch_index in range(0, batch_size, skip_window * 2):  # fill the batch inputs
            context = data[ci - skip_window:ci + skip_window + 1]
            # remove the target from context words
            target = context.pop(skip_window)
            # context = random.sample(context, skip_window * 2)
            batch_inputs[batch_index:batch_index +
                         skip_window * 2] = context
            batch_labels[batch_index:batch_index + skip_window * 2] = target
            ci += 1

        for rnn_i in range(batch_size):
            a, b = batch_inputs[rnn_i], batch_labels[rnn_i]
            context_word = '&' + int2word[a] + '&'
            target_word = int2word[b] + '&'
            context_vec = word2vec(char2int, context_word, n_chars)
            target_vec = word2vec(char2int, target_word, n_chars)
            rnn_inputs[rnn_i] = context_vec
            rnn_outputs[rnn_i] = target_vec

        if len(data) - ci - skip_window < batch_size:
            ci = skip_window
        batch_labels = batch_labels.reshape((-1, 1))
        yield batch_inputs, batch_labels, rnn_inputs, rnn_outputs


def subsampling(int_words, threshold=1e-5):
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {
        word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
    train_words = [word for word in int_words if random.random()
                   < (1 - p_drop[word])]
    return train_words


def evaluate(word2int, embeddings, corpus='data/news.txt', analogy='data/newan2.txt', embed_size=100):
    gensimw = GensimWrapper(file=corpus, test_file=analogy,
                            embed_size=embed_size, iter=0, log=False)
    gensimw.set_embeddings(word2int, embeddings)
    result = gensimw.evaluate()
    return result


def load_to_gensim(word2int, embeddings, corpus='data/news.txt', embed_size=100):
    gensimw = GensimWrapper(file=corpus, test_file=None,
                            embed_size=embed_size, iter=0, log=False)
    gensimw.set_embeddings(word2int, embeddings)
    return gensimw.model


def generate_for_char_langauge(words, int_words, int2word, char2tup,
                               batch_size=100, n_chars=13, n_consonant=40,
                               n_vowels=10, seq_length=5):
    """

    returns cbow input of integer sequence of words. the inputs are for RNN where the context, is normal
    but for the decoder, there is an input and output

    """
    decoder_output, decoder_input = {}, {}
    for index in int2word.keys():
        target = int2word[index]  # + '|'
        decoder_input[index] = '&' + target + '|'
        decoder_output[index] = target
    ci = 0
    while True:
        batch_inputs = np.ndarray((batch_size, seq_length), dtype=np.int32)
        batch_cons_output = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_output = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        batch_cons_dec_inputs = np.ndarray(
            (batch_size, n_chars, n_consonant), dtype=np.float32)
        batch_vow_dec_inputs = np.ndarray(
            (batch_size, n_chars,  n_vowels), dtype=np.float32)
        for i in range(batch_size):
            seq, ci = get_context_words(int_words, ci, seq_length + 1)
            target = seq[-1]
            batch_inputs[i] = np.array(seq[:seq_length])
            input_con, input_vow = word2vec_seperated(char2tup,
                                                      decoder_input[target], n_chars, n_consonant, n_vowels)
            output_con, output_vow = word2vec_seperated(char2tup,
                                                        decoder_output[target], n_chars, n_consonant, n_vowels)
            batch_cons_output[i] = output_con
            batch_vow_output[i] = output_vow

            batch_cons_dec_inputs[i] = input_con
            batch_vow_dec_inputs[i] = input_vow
            ci += 1
        yield [batch_inputs, batch_cons_dec_inputs, batch_vow_dec_inputs], [batch_cons_output, batch_vow_output]


def gen_imag_neg(data, skip_window, batch_size,
                 int2word, char2tup, neg_dest, n_chars, n_consonant, n_vowels):
    win_size = skip_window  # np.random.randint(1, skip_window + 1)
    i = win_size
    mat_width = n_consonant + n_vowels
    batch_y = np.zeros((batch_size, 1))
    while True:
        batch_input = []
        batch_output = []
        batch_neg = []
        for bi in range(0, batch_size, skip_window * 2):
            context = data[i - win_size: i + win_size + 1]
            target = context.pop(win_size)
            targets = [target] * (win_size * 2)
            negs = random.sample(neg_dest, win_size*2)
            # negs = targets
            con_mat, vow_mat = word2vec_seperated(char2tup,
                                                  int2word[target], n_chars, n_consonant, n_vowels)
            word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
                (1, n_chars, mat_width, 1))
            batch_input.extend([word_mat] * (win_size * 2))

            for cntx in context:
                con_mat, vow_mat = word2vec_seperated(char2tup,
                                                      int2word[cntx], n_chars, n_consonant, n_vowels)
                word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
                    (1, n_chars, mat_width, 1))
                batch_output.append(word_mat)

            for cntx in negs:
                con_mat, vow_mat = word2vec_seperated(char2tup,
                                                      int2word[cntx], n_chars, n_consonant, n_vowels)
                word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
                    (1, n_chars, mat_width, 1))
                batch_neg.append(word_mat)

            i += 1
            if i + win_size + 1 > len(data):
                i = win_size
        batch_input = np.vstack(batch_input)
        batch_output = np.vstack(batch_output)
        batch_neg = np.vstack(batch_neg)
        yield [batch_input, batch_output, batch_neg], batch_y

words = read_file()
vocab, word2int, int2word = build_vocab(words)
int_words = words_to_ints(word2int, words)
word2freq = get_frequency(words, word2int, int2word)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
ns_unigrams = ns_sample(word2freq, word2int, int2word, .75)
n_chars = 11 + 2
n_features = len(char2int)
batch_size = 120
embed_size = 128
skip_window = 5

# ins = word2vec_single(char2tup, ['ልዮ', 'ነው', 'ማለት'], 5, n_consonant, n_vowel)
# print(ins)
# gen = generate_word_images_multi(words, char2tup, batch_size, n_consonant, n_vowel)
# [x1, x2, x3], [y1, y2] = next(gen)
# print(x1.shape, x2.shape, x3.shape, y1.shape, y2.shape)
