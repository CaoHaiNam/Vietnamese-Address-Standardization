from __future__ import absolute_import

from .utils.parameters import *
from .utils.utils import contains_Vietchar, no_accent_vietnamese, tokenize
from nltk import ngrams

import re

import numpy as np
def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
	""" levenshtein_ratio_and_distance:
		Calculates levenshtein distance between two strings.
		If ratio_calc = True, the function computes the
		levenshtein distance ratio of similarity between two strings
		For all i and j, distance[i,j] will contain the Levenshtein
		distance between the first i characters of s and the
		first j characters of t
	"""
	rows = len(s)+1
	cols = len(t)+1
	distance = np.zeros((rows,cols),dtype = int)

	for i in range(1, rows):
		for k in range(1,cols):
			distance[i][0] = i
			distance[0][k] = k
  
	for col in range(1, cols):
		for row in range(1, rows):
			if s[row-1] == t[col-1]:
				cost = 0 
			else:
				if ratio_calc == True:
					cost = 2
				else:
					cost = 1
			distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
								 distance[row][col-1] + 1,          # Cost of insertions
								 distance[row-1][col-1] + cost)     # Cost of substitutions
	if ratio_calc == True:
		Ratio = ((len(s)+len(t)) - distance[row][col]) / float(len(s)+len(t))
		return Ratio
	else:
		return distance[row][col]

def get_ngrams(text, n):
	n_grams = ngrams(text, n)
	return [''.join(grams) for grams in n_grams]



def jaccard_similarity(string1, string2):
	sum = 0
	n_gram = 3
	list1 = get_ngrams(re.sub(r'[^\w\s]', '', string1.lower()).strip(), n_gram);
	list2 = get_ngrams(re.sub(r'[^\w\s]', '', string2.lower()).strip(), n_gram);
	intersection = len(list(set(list1).intersection(list2)))
	union = (len(list1) + len(list2)) - intersection
	if union == 0:
		return float(0)
	sum += float(intersection / union)
	return float(sum)

def c_score(string1, string2):
	list2 = string2.split(", ")
	c = 0
	for i in list2:
		if i in string1:
			c += len(i.split(" "))
	return 0

def tokenize_field(name, field):
	if field == 'district':
		field = 'DIST'
	else:
		field = field.upper()
	words = tokenize(name)
	labels = []
	if (len(words)) != 0:
		labels.append('B_' + field)
	for _ in range(len(words)-1):
		labels.append('I_' + field)
	return words, labels


def extract_features(raw_add, entities, candidate):
	features = {}
	#Bias
	features.update({'bias': 1})
	#Lexical feature

	#Admin_level in crf
	crf_max_lv = 0
	for entity, field, loc in entities:
		features.update({'crf:{}:lv'.format(field) :  1})
		crf_max_lv = max(crf_max_lv, MAP_LEVEL[field])

	# features.update({'crf_max_lv' : crf_max_lv})
	#Admin_level in candidate
	cdd_max_lv = 0
	for field in FIELDS:
		if field in candidate.keys():
			features.update({'cdd:{}:lv'.format(field) : 1})
			cdd_max_lv = max(cdd_max_lv, MAP_LEVEL[field])

	# features.update({'cdd_max_lv' : cdd_max_lv})

	# features.update({'diff_lv': abs(cdd_max_lv - crf_max_lv)})
	cvc = contains_Vietchar(raw_add)
	# Elastic Score
	for field in FIELDS:
		if field + '_score' in candidate.keys():
			features.update({'el:{}:s'.format(field) : float(candidate[field+'_score'])} )
	# value = 0
	# for field in FIELDS:
	# 	if field + '_score' in candidate:
	# 		value += candidate[field + '_score']
	# features.update({'elastic:score': value})

	#min admin level 
	min_field_cdd = ''
	for field in FIELDS:
		if field in candidate:
			min_field_cdd = field

	#other score
	matched_entities = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	if cvc == True:
		#Is contain vietnamese character
		features.update({'isVietnamese': 1})
		#Entity Score
		for entity, label, loc in entities:
			for field in FIELDS:
				if field not in candidate:
					continue
				value = 1 if entity.lower() == candidate[field] else 0
				if field == label:
					matched_entities[field] = value
				features.update({'{}:{}:{}:{}'.format(loc, label, field, 'en'): value})

		#Jaccard Score
		for entity, label, loc in entities:
			for field in FIELDS:
				if field not in candidate:
					continue
				value = jaccard_similarity(entity, candidate[field])
				# if field == label:
				features.update({'{}:{}:{}:{}'.format(loc, label, field, 'jc'): value})

		#Levenshtein Score
		for entity, label, loc in entities:
			for field in FIELDS:
				if field not in candidate:
					continue
				value = levenshtein_ratio_and_distance(entity.lower(), candidate[field].lower())
				# if field == label:
				features.update({'{}:{}:{}:{}'.format(loc, label, field, 'lvt'): value})
		
		if min_field_cdd != '':
			if matched_entities[min_field_cdd] == 0:
				if candidate[min_field_cdd + '_score'] == 0:
					features.update({'lost:min_lv': 1})
				else:
					# features.update({'els_cdd:min_lv': candidate[min_field_cdd + '_score']})
					for entity, label, loc in entities:
						value = 1 if entity.lower() == candidate[min_field_cdd] else 0
						features.update({'rep_min:{}:{}'.format(label,min_field_cdd): value})
						value = 1 if candidate[min_field_cdd] in entity.lower() else 0
						features.update({'rep_min:{}:{}:in'.format(label,min_field_cdd): value})

	else:
		#Entity Score with no_accent_vietnamese
		for entity, label, loc in entities:
			for field in FIELDS:
				if field not in candidate:
					continue
				value = 1 if no_accent_vietnamese(entity.lower()) == no_accent_vietnamese(candidate[field].lower()) else 0
				if field == label:
					matched_entities[field] = value
				features.update({'{}:{}:{}:{}:{}'.format(loc, label, field, 'en', 'nav'): value})
				
		#Jaccard Score with no_accent_vietnamese
		for entity, label, loc in entities:
			for field in FIELDS:
				if field not in candidate:
					continue
				value = jaccard_similarity(no_accent_vietnamese(entity.lower()), no_accent_vietnamese(candidate[field].lower()))
				# if field == label:
				features.update({'{}:{}:{}:{}:{}'.format(loc, label, field, 'jc', 'nav'): value})
		#Levenshtein Score with no_accent_vietnamese
		for entity, label, loc in entities:
			for field in FIELDS:
				if field not in candidate:
					continue
				value = levenshtein_ratio_and_distance(no_accent_vietnamese(entity.lower()), no_accent_vietnamese(candidate[field].lower()))
				# if field == label:
				features.update({'{}:{}:{}:{}:{}'.format(loc, label, field, 'lvt', 'nav'): value})

		if min_field_cdd != '':
			if matched_entities[min_field_cdd] == 0:
				if candidate[min_field_cdd + '_score'] == 0:
					features.update({'lost:min_lv': 1})
				else:
					# features.update({'els_cdd:min_lv': candidate[min_field_cdd + '_score']})
					for entity, label, loc in entities:
						value = 1 if no_accent_vietnamese(entity.lower()) == no_accent_vietnamese(candidate[min_field_cdd].lower()) else 0
						features.update({'rep_min:{}:{}'.format(label,min_field_cdd): value})
						value = 1 if no_accent_vietnamese(candidate[min_field_cdd].lower()) in no_accent_vietnamese(entity.lower()) else 0
						features.update({'rep_min:{}:{}:in'.format(label,min_field_cdd): value})
				
	return features