# -*- coding: utf-8 -*-
import string
from random import randint
import re
import operator
import pickle

def star_unique_file(input_filename, output_filename):
	input_file = open(input_filename, 'r', encoding='UTF-8', newline='')
	file_contents = input_file.read()
	input_file.close()
	duplicates = []
	word_list = file_contents.split()
	file = open(output_filename, 'w', encoding='UTF-8', newline='')
	for word in word_list:
		if "●" in word:
			file.write(str(word) + " ")
	file.close()

star_unique_file("freq_data/letters_with_dots.txt", "freq_data/test_words.txt")

star_words=[]
with open('freq_data/test_words.txt', 'r', encoding='UTF-8', newline='') as myfile:
	data=myfile.read()
	star_words = data.split()

unique_words=[]
with open('freq_data/letters_only.txt', 'r', encoding='UTF-8', newline='') as myfile:
	data=myfile.read()
	unique_words = data.split() 

temp1=""
temp2=""
temp4 = []
checker=0
index=0
correct=0
confidentcounter=0
dots=[]
for i in star_words:
	print("##" * 33)
	checker=0
	predictions={}
	check_one=0
	for j in unique_words:
		if len(i)==len(j):
			for x in range(len(i)):
				if i[x]==j[x]:
					check_one=1
					break
			if(len(i)==1):
				check_one=1
			if check_one==1:
				dots=[m.start() for m in re.finditer('●', i)]
				temp1=i.replace("●","")
				temp2=j
				temp3=""
				counter=0
				for x in dots:
					temp2 = temp2[:(x-counter)] + temp2[(x + 1-counter):]
					counter+=1
				if temp1==temp2:
					checker=1
					if i+" "+j+" " in predictions:
						predictions[i+" "+j+" "] += 1
					else:
						predictions[i+" "+j+" "]=1
	if checker==1:
		maximum=max(predictions, key=predictions.get)
		temp4.append(maximum.split(" ")[1])
		print("Dotted and Prediction: "+maximum)
		if predictions[maximum]/sum(predictions.values())>.9:
			confidentcounter+=1
		print("Max Frequency:", predictions[maximum]," Percentage frequency over matches:",str(round(100*predictions[maximum]/sum(predictions.values()),2))+"%")
		correct+=1
	if checker==0:
		temp4.append(i)
		print("Not Found:",i)		
print("##" * 33)
print("Percentages Matches Found:"+str(100*correct/len(star_words))+"%")
print("Confident Frequencies (Over .75): "+str(100*confidentcounter/correct)+"%")

dot_file=open('freq_data/predicted', 'wb')
pickle.dump(temp4, dot_file)
dot_file.close()
