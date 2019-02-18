import sys
from nltk.stem import SnowballStemmer as Stemmer

if len(sys.argv) != 3:
	print('Usage error: prog.py [training data file] [test data file]')
	exit

# Hold all distinct words between the files
vocabulary = set()

# NLTK stemmer to normalize words
stemmer = Stemmer('english')

print('Opening training data file')
with open(sys.argv[1]) as f:
	trainlines = f.readlines()

print('%i lines of training data found' % len(trainlines))
for line in trainlines:
	words = line.split()
	subject = words[0]
	for word in words:
		if len(word) > 2:
			word = stemmer.stem(word)
			vocabulary.add(word)

print('Opening test data file')
with open(sys.argv[2]) as f:
	testlines = f.readlines()

print('%i lines of test data found' % len(testlines))
for line in testlines:
	words = line.split()
	for word in words:
		if len(word) > 2:
			word = stemmer.stem(word)
			vocabulary.add(word)

print('%i distinct words total found' % len(vocabulary))