# -*- coding: utf-8 -*-
from random import randint
import pickle
import string
import numpy as np
import argparse
import threading
import word_corpus_data as data
import re

################################################################
# Options
################################################################

parser = argparse.ArgumentParser(description='Data Generator')
parser.add_argument('--asterisk', type=bool, default=True,
                    help='Generate asterisked PTB corpus?')
parser.add_argument('--convert', type=bool, default=True,
                    help='Convert asterisked PTB corpus to one np array?')
parser.add_argument('--split', type=bool, default=True,
                    help='Split asterisked PTB corpus to train_data and val_data?')
parser.add_argument('--train', type=bool, default=True,
                    help='Convert train_data to np array?')
parser.add_argument('--val', type=bool, default=True,
                    help='Convert val_data to np array?')
args = parser.parse_args()

#################################################################
# Clean and asterisk PTB corpus, generate test data
#################################################################

corpus = data.Corpus('word_data')
t_array = corpus.train
t_array = t_array.astype(np.int64)

if args.asterisk:
    with open('word_data/old_books.txt', 'r', encoding='UTF-8', newline='') as myfile:
        data = myfile.read().replace('\n', '')

    data2 = corpus.test
    i = 0
    asterisks2 = []

    for i in range(0, len(data2) - 1):
        word = corpus.dictionary.idx2word[int(data2[i])]        
        pattern = re.compile(r'●')
        text_list = (pattern.findall(word))

        if len(text_list) >= 1:
            asterisks2.append([i, data2[i]])

    test_data=open('word_data/test_data', 'wb')
    pickle.dump(asterisks2, test_data)
    test_data.close()

    data3 = corpus.train
    data3 = data3.astype(np.int64)
    data3 = data3[len(data3)//10:]

    with open('word_data/train_data_array', 'wb') as handle:
        pickle.dump(data3, handle, protocol=pickle.HIGHEST_PROTOCOL)

    data4 = corpus.train
    data4 = data4.astype(np.int64)
    data4 = data4[:len(data4)//10]
    with open('word_data/val_data_array', 'wb') as handle:
        pickle.dump(data4, handle, protocol=pickle.HIGHEST_PROTOCOL)
