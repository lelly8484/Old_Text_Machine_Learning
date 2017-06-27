import xml.etree.ElementTree as ET
import string

file=open("./data/oldtext.txt","w+")
wordcount=[]


counter=0
i=0
def parser(node):
	if node.findall('*')!=None:
		global i
		for child in node.findall('*'):
			if (child.tag=='{http://www.tei-c.org/ns/1.0}w' or child.tag=='{http://www.tei-c.org/ns/1.0}pc'):
				if child.text!=None:
					file.write(child.text)
					if child.tag=='{http://www.tei-c.org/ns/1.0}w':
						i=i+1
			# elif (child.tag=='{http://www.tei-c.org/ns/1.0}date' and i<1)
			# 	file.write(child.text)
			# 	i=1
			elif (child.tag=='{http://www.tei-c.org/ns/1.0}c'):
				file.write(" ")
			else:
				parser(child)

datalist=['./EEBO1640-60/A00/A00011.xml','./EEBO1640-60/A00/A00214.xml',
'./EEBO1640-60/A00/A00289.xml','./EEBO1640-60/A00/A00293.xml',
'./EEBO1640-60/A00/A00395.xml'
,'./EEBO1640-60/A01/A01084.xml',
'./EEBO1640-60/A01/A01210.xml',
'./EEBO1640-60/A01/A01344.xml',
'./EEBO1640-60/A01/A01531.xml',
'./EEBO1640-60/A01/A01750.xml']
# './EEBO1640-60/A01/A01773.xml',
# './EEBO1640-60/A01/A01775.xml',
# './EEBO1640-60/A01/A01779.xml',
# './EEBO1640-60/A01/A01989.xml','./EEBO1640-60/A02/A02199.xml',
# './EEBO1640-60/A02/A02262.xml','./EEBO1640-60/A02/A02455.xml',
# './EEBO1640-60/A02/A02549.xml','./EEBO1640-60/A02/A02755.xml','./EEBO1640-60/A02/A02769.xml'


for item in datalist:
	parser(ET.parse(item).getroot())
	wordcount.append(i)

print (wordcount)

length=[]
middle=[]
ranges=[]
for i in range(0,len(wordcount)):
	if i==0:
		length.append(wordcount[0])
	else:
		length.append(wordcount[i]-wordcount[i-1])

for i in range(0,len(wordcount)):
	if i==0:
		middle.append(round(wordcount[0]/2))
	else:
		middle.append(round(wordcount[i]-(length[i]/2)))


for i in range(0,len(wordcount)):
	ranges.append(middle[i])
	ranges.append(round(middle[i]+(.1*length[i])))

print (ranges)

file.close()

file2=open("./data/tenth_oldtext.txt","w")

file1=open("./data/oldtext.txt","r")

all_words=[]
for word in file1:
	all_words = word.split()
print (len(all_words))
wantedtenth=[]
myycounter=0

print (str(ranges[myycounter]))
for i in range(0,len(length)-1):
	wantedwords=all_words[ranges[myycounter]:ranges[myycounter+1]]
	file2.write(" ".join(wantedwords))
	myycounter=myycounter+2






# all_letters = string.ascii_letters+string.punctuation+string.digits+string.whitespace+" 〈◊〉〈…〉▪•*§‡¶†☞☜…— ÁáÀàÅåÄäāâÇçÉéÈèÊêëēÍíÌìÎîÑñÓóÒòÔôÖöōÚúÙùÜüūŽž  "

# file.close()
# i=0
# with open("./data/oldtext.txt") as f:
# 	c=f.read(1372724)
# 	print(c)
# 	while True:
# 		c=f.read(1)
# 		i=i+1
# 		if not c:
# 			print ("End of file")
# 			break
# 		if c in all_letters:
			
# 			v=c
# 			continue
# 		else:
# 			print (v,c,i)


# TEi->text->body->div