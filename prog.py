import sys
import math
from nltk.stem import SnowballStemmer as Stemmer
from nltk.corpus import stopwords


# Use Naive Bayes classifier
#
# C = max(P(c) ∏ P(w | c))
def calcClass(words):
	# Preformat words
	for word in words:
		if len(word) > 2:
			word = stemmer.stem(word)
		else:
			words.remove(word)
			
	# Track current best guess
	currMax = -sys.maxsize
	currSub = ''

	# Loop through each subject we learned about in training
	for subject in documents:
		# Start with probability of that subject based on number of documents with that subject
		temp = math.log10(documents[subject] / numDocuments)

		# Iterate through words
		for word in words:
			# Add the number of occurences of that word for documents with this subject + 1,
			# divided by the total number of words for that subject, plus the size of the vocabulary
			try:
				temp += math.log10((texts[subject][word] + 1) / (totals[subject] + vocabularySize))
			except KeyError:
				temp += math.log10(1 / (totals[subject] + vocabularySize))

		if temp > currMax:
			currMax = temp
			currSub = subject

	return currSub


# Program entry
if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('Usage error: prog.py [training data file] [test data file]')
		exit

	# Set to hold all distinct words between the files
	vocabulary = set()

	# Dictionary to hold counts of each subject
	documents = {}

	# Dictionary to hold total number of words for each subject
	totals = {}

	# Dictionary to associate doc class (key) with a nested dictionary (value). This internal
	# dictionary will associate each word that occurs (key) with the number of times
	# it has occurred (value)
	texts = {}

	# NLTK stemmer to normalize words
	stemmer = Stemmer('english')

	# NLTK set of stopwords too common to provide context
	stops = stopwords.words('english')

	print('Opening training data file')
	with open(sys.argv[1]) as f:
		print('Reading training data file')
		trainlines = f.readlines()
	numDocuments = len(trainlines)

	print('%i lines of training data found' % len(trainlines))
	for line in trainlines:
		words = line.split()
		words = [word for word in words if word not in stops]
		subject = words[0]

		if subject not in documents:
			documents[subject] = 1
		else:
			documents[subject] += 1

		if subject not in totals:
			totals[subject] = 0

		if subject not in texts:
			texts[subject] = {}

		for word in words[1:]:
			# Only include words greater than 2 letters, after being stemmed
			if len(word) > 2:
				word = stemmer.stem(word)
				totals[subject] += 1
				vocabulary.add(word)
				if word not in texts[subject]:
					texts[subject][word] = 1
				else:
					texts[subject][word] += 1

	# Also used in Bayes equation
	vocabularySize = len(vocabulary)
	print('%i distinct words found in the training data' % vocabularySize)


	print('Opening test data file')
	with open(sys.argv[2]) as f:
		print('Reading test data file')
		testlines = f.readlines()

	print('%i lines of test data found' % len(testlines))
	print('Analyzing documents...')
	correctClassifications = 0
	for line in testlines:
		words = line.split()
		words = [word for word in words if word not in stops]
		# Actual subject stored as first word of each entry
		if calcClass(words[1:]) == words[0]:
			correctClassifications += 1

	print('%i correct classifications out of %i entries, or %.2f%% correct' % (correctClassifications, len(testlines), float(100 * correctClassifications/len(testlines))))