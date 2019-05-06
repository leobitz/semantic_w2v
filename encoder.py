from data_handle import *

words = read_file()
vocab, word2int, int2word = build_vocab(words)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
vowels = ['e', 'u', 'i', 'a', 'ê', 'æ', 'o', 'õ', 'ø', 'ü']

new_file_name = ""

def encode(word):
    chars = []
    if word in '*,.,/,|,#,U,N,K,&, ':
        return word
    for char in word:
        if char in '*,.,/,|,#,U,N,K,&, ':
            chars += [char]
            continue
        c, v = char2tup[char]
        new_v = vowels[v]
        tup = "{0}-5".format(c)
        if tup not in tup2char:
            tup = "{0}-1".format(c)
        new_c = tup2char[tup]
        if c == 5:
            chars += [new_c]
        else:
            chars += [new_c, new_v]
    return ''.join(chars)


word_dict = {}

for word in words:
    if word not in word_dict:
        word_dict[word] = encode(word)

# lines = open('data/newan2.txt', encoding='utf-8').readlines()
# new_lines = []
# for line in lines:
#     if ':' in line:
#         new_lines.append(line[:-1])
#         continue
#     ws = line[:-1].split(' ')
#     new_words = []
#     for w in ws:
#         new_words.append(encode(w))
#     new_line = ' '.join(new_words)
#     if new_line not in new_lines:
#         new_lines.append(new_line)
# final_text = '\n'.join(new_lines)
# open('data/new_ana2.txt', mode='w', encoding='utf-8').write(final_text)

# lines = open('data/anomaly.txt', encoding='utf-8').readlines()
# new_lines = []
# for line in lines:
#     if ':' in line:
#         new_lines.append(line[:-1])
#         continue
#     ws = line[:-1].split(' ')
#     new_words = []
#     for w in ws[:-1]:
#         new_words.append(encode(w))
#     new_words.append(ws[-1])
#     new_line = ' '.join(new_words)
#     if new_line not in new_lines:
#         new_lines.append(new_line)
# final_text = '\n'.join(new_lines)
# open('data/anomaly_new.txt', mode='w', encoding='utf-8').write(final_text)

new_text = []
for word in words:
    new_text.append(word_dict[word])
new_text = ' '.join(new_text)
open('data/news_en2.txt', mode='w', encoding='utf-8').write(new_text)
