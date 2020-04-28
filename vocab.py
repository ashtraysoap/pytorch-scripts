import torch

class Vocab():
    def __init__(self):
        self.word2index = {"SOS": 0, "EOS": 1, "PAD": 2, "UNK": 3}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3: "UNK"}
        self.n_words = 4

    def build_vocab(self, sentences, max_size=None):
        if max_size is None:
            for s in sentences:
                self.add_sentence(s)
            return
        
        w2c = {}
        for s in sentences:
            for w in s.split(' '):
                if w not in w2c:
                    w2c[w] = 1
                else:
                    w2c[w] += 1
        sorted_w2c = sorted(w2c.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(max_size, len(w2c))):
            w = sorted_w2c[i][0]
            c = sorted_w2c[i][1]
            self.word2index[w] = self.n_words
            self.word2count[w] = c
            self.index2word[self.n_words] = w
            self.n_words += 1


    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_to_tensor(self, s, max_seq_len=None):
        s = s.split(' ')
        s = map(lambda x: x if x in self.word2index else "UNK", s)
        toks = [self.word2index[w] for w in s]
        toks.append(self.word2index["EOS"])
        
        if max_seq_len is not None:
            t = [self.word2index["PAD"]] * max_seq_len
            for i in range(min(len(toks), max_seq_len)):
                t[i] = toks[i]
            toks = t
        toks = [[x] for x in toks]
        return torch.tensor(toks)

    def tensor_to_sentence(self, tensor):
        tensor = tensor.view(-1)
        s = [self.index2word[i.item()] for i in tensor]
        return ' '.join(s)


    def save(self, path='vocab.txt'):
        with open(path, mode='w', encoding='utf-8') as outf:
            for i in range(self.n_words):
                w = self.index2word[i]
                if w not in ["SOS", "EOS", "PAD", "UNK"]:
                    l = w + '\t' + str(self.word2count[w]) + '\n'
                    outf.write(l)

    def load(self, path='vocab.txt'):
        vocab = open(path, mode='r', encoding='utf-8').read().strip().split('\n')
        for e in vocab:
            e = e.split('\t')
            w = e[0]
            wc = e[1]
            self.index2word[self.n_words] = w
            self.word2index[w] = self.n_words
            self.word2count[w] = wc
            self.n_words += 1
