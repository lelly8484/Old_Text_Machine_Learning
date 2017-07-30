import os
import torch
import numpy as np
import re # added

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'old_books.txt'))
        self.valid = self.tokenize(os.path.join(path, 'old_books.txt'))
        self.test = self.tokenize2(os.path.join(path, 'old_books.txt'))
    def tokenize(self, path):
        assert os.path.exists(path)

        with open(path, 'r', encoding='UTF-8', newline='') as f:
            tokens = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']

                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='UTF-8', newline='') as f:
            ids = np.array([]) # added
            token = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']

                for word in words:
                    ids = np.append(ids, self.dictionary.word2idx[word]) # added
                    token += 1

        return ids




    def tokenize2(self, path):
        assert os.path.exists(path)

        with open(path, 'r', encoding='UTF-8', newline='') as f:
            tokens = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']
                tokens += len(words)

                for word in words:
                    self.dictionary.add_word(word)


        # Tokenize file content
        with open(path, 'r', encoding='UTF-8', newline='') as f:
            ids = []
            token = 0
            for line in f:
                words = re.split("[,.:; ]", line)
                words += ['<eos>']
                                
                for word in words:
                    ids.append(str(self.dictionary.word2idx[word]))
                    token += 1

        return ids
