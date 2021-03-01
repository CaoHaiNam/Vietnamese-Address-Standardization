import pycrfsuite
import sys
import os
import codecs
import json
from Parameters import *

CRF_MODEL_FILE = './CRF/crf_rat_3.model'
RAT_DICT_FILE = './CRF/rat_dict.json'
PUNCTUATIONS=',.-()#|/\\'
USE_RAT = True

# utils
def ispunct(word):
    for char in word:
        if char not in PUNCTUATIONS:
            return 0
    return 1

# data_processing
def tokenize(inp):
	inp = inp.strip()
	words = []
	word = ''
	for char in inp:
		if char == ' ':
			if len(word) != 0:
				words.append(word)
				word = ''
		elif char in PUNCTUATIONS:
			if len(word) != 0:
				words.append(word)
				word = ''
			words.append(char)
		else:
			word += char
	if len(word) != 0:
		words.append(word)

	return words

def wrap_postag(words):
	ret = []
	for word in words:
		ret.append([word,'O'])
	return ret

#feature_extraction
rat_dict = None
def extend_rat_features(word, features):
    global rat_dict
    if rat_dict == None:
        with codecs.open(RAT_DICT_FILE, encoding='utf8', mode='r') as f:
            rat_dict = json.load(f)

    if word.lower() in rat_dict:
        new_features = []
        for name_feature, value in rat_dict[word.lower()].items():
            new_features.append('%s=%s' % (name_feature, str(value)))
        features.extend(new_features)


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.ispunct=%s' % ispunct(word)
    ]

    if USE_RAT is True:
        extend_rat_features(word, features)

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:word.ispunct=%s' % ispunct(word1)
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    if i > 1:
        word1 = doc[i-2][0] + doc[i-1][0]
        features.extend([
            '-2:word.lower=' + word1.lower(),
            '-2:word.istitle=%s' % word1.istitle(),
            '-2:word.isupper=%s' % word1.isupper(),
            '-2:word.isdigit=%s' % word1.isdigit(),
            '-2:word.ispunct=%s' % ispunct(word1)
        ])


    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:word.ispunct=%s' % ispunct(word1)
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

# A function for extracting features in documents
def extract_features(doc):
	return [word2features(doc, i) for i in range(len(doc))]

def tag(inp):
	words = tokenize(inp)
	doc = wrap_postag(words)

	x = extract_features(doc)
	tagger = pycrfsuite.Tagger()
	tagger.open(CRF_MODEL_FILE)

	return words,tagger.tag(x)

def detect_entity(inp, tokens=None, labels=None):
	if tokens == None or labels == None:
		tokens, labels = tag(inp)
	entities = []

	n = len(tokens)
	buff = ''
	lbuff = ''
	isEntity = False

	for i in range(n):
		if (labels[i][0] == 'I'):
			buff += ' ' + tokens[i]
		else:
			if isEntity == True:
				key = lbuff.lower() 
				entities.append(key)

			buff = tokens[i]
			if labels[i][0] == 'B':
				if labels[i] == 'B_DIST':
					lbuff = 'DISTRICT'
				elif labels[i] == 'B_PRO':
					lbuff = 'NAME'
				else:
					lbuff = labels[i][2:]
				isEntity = True
			else:
				lbuff = labels[i]
				isEntity = False

	if isEntity == True:
		key = lbuff.lower() 
		entities.append(key)


	return entities

def get_better_add_NoneType(noisy_add):
    choice_tag = ['B_STREET', 'I_STREET', 'B_WARD', 'I_WARD', 'B_DIST', 'I_DIST', 'B_CITY', 'I_CITY']
    x = tag(noisy_add)
    res = []
    for i in range(len(x[1])):
        if x[1][i] in choice_tag:
            res.append(x[0][i])
    if len(res) != 0:
        return (' '.join(res))
    else:
        return noisy_add

def get_better_add(noisy_add):
    choice_tag = ['B_STREET', 'I_STREET', 'B_WARD', 'I_WARD', 'B_DIST', 'I_DIST', 'B_CITY', 'I_CITY', 'STREET_TYPE', 'WARD_TYPE', 'DIST_TYPE', 'CITY_TYPE']
    x = tag(noisy_add)
    res = []
    for i in range(len(x[1])):
        if x[1][i] in choice_tag:
            res.append(x[0][i])
    if len(res) != 0:
        return (' '.join(res))
    else:
        return noisy_add

