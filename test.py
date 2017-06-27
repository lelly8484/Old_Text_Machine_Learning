from random import randint
import pickle
import string
import numpy as np
import argparse
import threading

with open('./data/oldtext.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')

with open('./data/oldtext_pre_asterisked', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)