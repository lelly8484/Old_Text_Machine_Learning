# -*- coding: utf-8 -*-

import re
import glob
import string


with open("word_data/old_books.txt", "w", encoding='UTF-8', newline='') as write_file:
	for filename in glob.glob('XML/*.xml'):
		with open(filename, "r", encoding='UTF-8', newline='') as read_file:
			first = True
			for line in read_file:
				if re.match(".*lemma=.*", line):
					word = line.split("</w>")[0].split(">")[1]
					if first == True:
						write_file.write(word)
						first = False
					else:
						write_file.write(" " + word)
				elif re.match(".*</pc>", line):
					word = line.split("</pc>")[0].split(">")[1]
					write_file.write(word)
