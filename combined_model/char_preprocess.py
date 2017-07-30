# -*- coding: utf-8 -*-
from random import randint
import pickle
import string
import numpy as np
import argparse
import threading

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

specials = "▪ ●〈◊〉̄íàé…§❧áùèìòúꝑÉ·ÿöóëôî¶‑—ηçć ü⸫âûꝰêΩ⁎✚˙ΙΗΣΟΥ⸪“”☞ȧ†❀½¾⅓¼⅔⅛⅚⅝⅖⅗⅕ʒï☉īĕăōēĭŏāūŭꝓΨ‡ę⅘ȝπβÔÁΕΝΤΚἘΠΜΛΔÎΑ‖ě⸬κατῇλθενἐισἁδουΦχρꝙ♈♉♋♍♏♑♓♒♐♎♌♊♄♃♂♀☿☽☌⚹□△☍☊℥ωŕ☐ª╌ÇĈיוהבארäÓß☜Γº⁂חȣφγζμξψ♮⅙℈ŷέςś∵ΈΡΊ℞Βṗ☟☝ƿΘↂↁↈ£ńϹ×ꝭåÖ𝄢𝄡𝄞톼텮톺텥톹𝆹´άόῷÒÀþðꝧÈ☧Χ̇משכלדȜİ°℟℣∶⅞−ůËñῳėċżÙÛ∣☋⁙′ὈήῶῥὴꝯΞ″ꝗ⋆גזטךםנןסעפףצץקת¦ὸὶ̔ͅ⊙ṙǵ◆ÐṅÆἙ‘ΖŚÞźὅ𝄁ῦίἩὁἀ♡∷‴ὑ○♁🜕∝⅜ΌῚᾺΆ√‐✴Ú⌊∞¿Ύ▵◬🜹ἹὰÜÑŵ∴˜❍¯˘🜂✝🜔𝆶𝆷𝆸텯𝇋𝇍𝇈♯⋮☾🜄ὙἈ̂★ĉňƳƴ¡АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЮŝǽ⁏ЫЬἜ’Â🜍ↃÌ÷∽𝇊õ♥♭⊕⊗☼ȯύÏὨẏṫ±∓∼ἰđČýḃḅ؛🜖ĥø․⁁∠ὼǔḍ¨▴–▿ʹϊ̈ↇϛϟϡ͵🝆  " # ,.;\'\\1234567890&*-/"
include = "▪〈◊〉̄…§❧ꝑ·¶‑—⸫ꝰΩ⁎✚˙⸪“”☞†❀½¾⅓¼⅔⅛⅚⅝⅖⅗⅕ʒï☉ꝓΨ‡⅘ȝπΠΛΔ‖⸬τῇλθενἐισἁδΦꝙ♈♉♋♍♏♑♓♒♐♎♌♊♄♃♂♀☿☽☌⚹□△☍☊℥ωŕ☐ª╌יוהבאר☜Γº⁂חȣφγζμξψ♮⅙℈ŷέςś∵☟☝ƿΘↂↁↈ£ńϹ×ꝭ𝄢𝄡𝄞톼텮톺텥톹𝆹´ðמשכלדȜİ°℟℣∶⅞−∣☋⁙′ήΞ″ꝗ⋆גזטךםנןסעפףצץקת¦ὸὶ̔ͅ⊙◆Æ‘𝄁♡∷‴♁🜕∝⅜‐✴⌊∞¿▵◬🜹∴˜❍¯˘🜂✝🜔𝆶𝆷𝆸텯𝇋𝇍𝇈♯⋮☾🜄ὙἈ̂★БбГгЖжЗзЛлФфЦцЧчШшЩщЮŝǽ⁏ЫЬ’🜍Ↄ÷∽𝇊õ♥♭⊕⊗☼ȯύÏὨẏṫ±∓∼؛🜖ø․⁁∠¨▴–▿ʹϊ̈ↇϛϟϡ͵🝆"

all_letters = string.ascii_letters+string.punctuation+string.digits+specials
n_letters = len(all_letters)

if args.asterisk:
    with open('char_data/old_books.txt', 'r', encoding='UTF-8', newline='') as myfile:
        data = myfile.read().replace('\n', '')

    for i in specials:
        if i in include:
            data = data.replace(i, "")

    data = data.replace("ā", "ā")
    data = data.replace("v̄", "v")
    data = data.replace("n̄", "n")
    data = data.replace("ō", "ō")
    data = data.replace("ū", "ū")
    data = data.replace("ē", "ē")
    data = data.replace("h̄", "h")

    oldtext=open('char_data/oldtext_pre_asterisked', 'wb')
    pickle.dump(data, oldtext)
    oldtext.close()

    asterisks = []
    li = list(data)
    i = 0
    while i < len(data)-1:
        index = i
        i += 1
        if data[index] in "●":
            asterisks.append([index, data[index]])
            li[index] = '*'          
    data = ''.join(li)

    oldtext_asterisked=open('char_data/oldtext_asterisked', 'wb')
    pickle.dump(data, oldtext_asterisked)
    oldtext_asterisked.close()

    test_data=open('char_data/test_data', 'wb')
    pickle.dump(asterisks, test_data)
    test_data.close()

##################################################################
# Convert string of PTB corpus to numpy array of embedding
# Use multi-threading (8)
##################################################################
def char_index(char):
    return all_letters.index(char)


class MyThread (threading.Thread):
    def __init__(self, id, data, data_type):
        threading.Thread.__init__(self)
        self.id = id
        self.data = data
        self.data_type = data_type

    def run(self):
        data_array = np.array([])
        for i in range(len(self.data) // 8 * self.id, min(len(self.data) // 8 * (self.id + 1), len(self.data))):
            data_array = np.append(data_array, char_index(self.data[i]))
            if i % (len(self.data) // 50) == 0:
                print("Thread {} at {:2.1f}%".format(self.id, 100 * (i - len(self.data) // 8 * self.id) /
                      (min(len(self.data) // 8 * (self.id + 1), len(self.data)) - len(self.data) // 8 * self.id)))
        with open('char_data/{}_data_array_{}'.format(self.data_type, self.id), 'wb') as handle:
            pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def embed(data, data_type):
    thread0 = MyThread(0, data, data_type)
    thread1 = MyThread(1, data, data_type)
    thread2 = MyThread(2, data, data_type)
    thread3 = MyThread(3, data, data_type)
    thread4 = MyThread(4, data, data_type)
    thread5 = MyThread(5, data, data_type)
    thread6 = MyThread(6, data, data_type)
    thread7 = MyThread(7, data, data_type)
    thread8 = MyThread(8, data, data_type)

    thread0.start()
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()

    thread0.join()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()
    thread8.join()

    with open('char_data/{}_data_array_0'.format(data_type), 'rb') as handle:
        data_array_0 = pickle.load(handle)

    with open('char_data/{}_data_array_1'.format(data_type), 'rb') as handle:
        data_array_1 = pickle.load(handle)

    with open('char_data/{}_data_array_2'.format(data_type), 'rb') as handle:
        data_array_2 = pickle.load(handle)

    with open('char_data/{}_data_array_3'.format(data_type), 'rb') as handle:
        data_array_3 = pickle.load(handle)

    with open('char_data/{}_data_array_4'.format(data_type), 'rb') as handle:
        data_array_4 = pickle.load(handle)

    with open('char_data/{}_data_array_5'.format(data_type), 'rb') as handle:
        data_array_5 = pickle.load(handle)

    with open('char_data/{}_data_array_6'.format(data_type), 'rb') as handle:
        data_array_6 = pickle.load(handle)

    with open('char_data/{}_data_array_7'.format(data_type), 'rb') as handle:
        data_array_7 = pickle.load(handle)

    with open('char_data/{}_data_array_8'.format(data_type), 'rb') as handle:
        data_array_8 = pickle.load(handle)

    # appending arrays of different sizes is problematic
    data_array = np.append(data_array_0, [data_array_1, data_array_2, data_array_3,
                                          data_array_4, data_array_5, data_array_6, data_array_7])

    data_array = np.append(data_array, data_array_8)

    with open('char_data/{}_data_array'.format(data_type), 'wb') as handle:
        pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


if args.convert:
    corpus=open('char_data/oldtext_asterisked', 'rb')
    corpus=pickle.load(corpus)
    embed(corpus, 'train')


##############################################################################
# Split PTB corpus into training and validation data
##############################################################################

if args.split:
    with open('char_data/oldtext_asterisked', 'rb') as handle:
        corpus = pickle.load(handle)
        
    nine = len(corpus) * 9 // 10
    train_data = corpus[:nine]
    val_data = corpus[nine:]

    with open('char_data/train_data', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('char_data/val_data', 'wb') as handle:
        pickle.dump(val_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


##############################################################################
# Convert training and validation data to array embeddings
##############################################################################

if args.train:
    with open('char_data/train_data', 'rb') as handle:
        train_data = pickle.load(handle)
    embed(train_data, 'train')

if args.val:
    with open('char_data/val_data', 'rb') as handle:
        val_data = pickle.load(handle)
    embed(val_data, 'val')
