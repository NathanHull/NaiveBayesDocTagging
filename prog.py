import sys
from nltk.stem import SnowballStemmer as Stemmer


# Use Naive Bayes classifier
# def calcClass(documents, totals, texts, words, sample):
def calcClass(words):
	currMax = 0
	currSub = ''
	
	for subject in documents:
		temp = documents[subject]
		for word in words:
			temp *= (texts[subject][word] + 1) / (totals[subject] + sample)

		if temp > currMax:
			currMax = temp
			currSub = subject

	return currSub


if len(sys.argv) != 3:
	print('Usage error: prog.py [training data file] [test data file]')
	exit

# Set to hold all distinct words between the files
vocabulary = set()

# Dictionary to hold counts of each subject
documents = {}

# Dictionary to hold total number of words for each subject
totals = {}

# Dictionary to associate doc class (key) with a dictionary
# (value). This internal dictionary will associate each
# word that occurs (key) with the number of times it has
# occurred (value)
texts = {}

# NLTK stemmer to normalize words
stemmer = Stemmer('english')

print('Opening training data file')
with open(sys.argv[1]) as f:
	trainlines = f.readlines()

print('%i lines of training data found' % len(trainlines))
for line in trainlines:
	words = line.split()
	subject = words[0]

	if subject not in documents:
		documents[subject] = 1
	else:
		documents[subject] = documents[subject] + 1

	if subject not in totals:
		totals[subject] = 0

	if subject not in texts:
		texts[subject] = {}

	for word in words:
		if len(word) > 2:
			word = stemmer.stem(word)
			totals[subject] += 1
			vocabulary.add(word)
			if word not in texts[subject]:
				texts[subject][word] = 1
			else:
				texts[subject][word] = texts[subject][word] + 1

print('%i distinct words found in the training data' % (len(vocabulary)))

print()

print('Opening test data file')
with open(sys.argv[2]) as f:
	testlines = f.readlines()

print('%i lines of test data found' % len(testlines))
print('Analyzing documents...')
correctClassifications = 0
for line in testlines:
	words = line.split()
	if calcClass(words) == words[0]:
		correctClassifications += 1

print('%i correct classifications out of %i entries, or %f%% correct' % (correctClassifications, len(testlines), float(correctClassifications/len(testlines))))