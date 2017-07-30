# -*- coding: utf-8 -*-

import re
import glob
import string
import pickle


with open('word_data/predicted', 'rb') as handle:
	dot_list = pickle.load(handle)

count = 0
for filename in glob.glob('freq_output/*.xml'):
	with open("word_output/" + filename.split("\\")[1].split(".")[0] + ".xml", "w", encoding='UTF-8', newline='') as write_file:
		with open(filename, "r", encoding='UTF-8', newline='') as read:
			for i in read:
				if re.match(".*lemma=.*", i):
					word = i.split("</w>")[0].split(">")[1]
					original = word
					text_list = (re.compile(r'●').findall(word))

					if len(text_list) > 0:
						word = word.replace(word, dot_list[count], 1)
						count += 1
						i = i.replace('>' + original + '</w>', '>' + word + '</w>')
					write_file.write(i)
				else:
					write_file.write(i)
