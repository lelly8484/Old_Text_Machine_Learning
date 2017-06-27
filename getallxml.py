# -*- coding: utf-8 -*-

import re
import glob
import string

data = []
dict_file = {}
all_letters = string.ascii_letters + " .,;'-"

write_file = open("old_books.txt", "w", encoding='UTF-8', newline='')

for filename in glob.glob('TCP-Phase1-1600/*/*/*.xml'):
	with open(filename, "r", encoding='UTF-8', newline='') as read_file:
		print (filename)
		# print (read_file)
		for line in read_file:
			if re.match(".*lemma=.*", line):
				word = line.split("</w>")[0].split(">")[1]
				write_file.write(word)
			elif re.match(".*</pc>", line):
				word = line.split("</pc>")[0].split(">")[1]
				write_file.write(word)
			elif re.match(".*<c> </c>.*", line):
				write_file.write(" ")
			elif re.match(".*<date>[0-9]*</date>", line):
				word = line.split("<date>")[1].split("</date>")[0]
				# write_file.write(word + "\n")

			# 	# exclude = ".*(•|§|‡|†|…|☜|☞|!|@|#|\$|%|\^|&|\*|\(|\)|_|\+|\\|\[|\]|\{|\}|[0-9]|\.).*"
			# 	if re.match("[^" + all_letters + "]", word):
			# 		continue
			# 	elif not re.match(".*•.*", word):
			# 		dict_file[word] = ""

write_file.close()
