# -*- coding: utf-8 -*-

import re
import glob
import string
import pickle


with open('char_data/predicted', 'rb') as handle:
	dot_list = pickle.load(handle)

count = 0
for filename in glob.glob('word_output/*.xml'):
	with open("FIXED_XML/" + filename.split("\\")[1].split(".")[0] + ".xml", "w", encoding='UTF-8', newline='') as write_file:
		with open(filename, "r", encoding='UTF-8', newline='') as read:
			for i in read:
				if re.match(".*lemma=.*", i):
					word = i.split("</w>")[0].split(">")[1]
					original = word
					text_list = (re.compile(r'●').findall(word))

					if len(text_list) > 0:
						for j in range(0, len(text_list)):
							word = word.replace('●', dot_list[count], 1)
							count += 1
						i = i.replace('>' + original + '</w>', ' type=\"machine-fixed\" corresp=\"' + original + '\">' + word + '</w>')

					write_file.write(i)
				else:
					write_file.write(i)
