from __future__ import absolute_import

from .utils.parameters import *

import codecs, json

def load_data(filename):
	data = []
	with codecs.open(filename, encoding='utf8', mode='r') as f:
		sample_list = json.load(f)['data']
		for sample in sample_list:
			data.append((sample['noisy_add'],sample['std_add'], sample['id']))
	return data

def preprocess(raw_data):
	ret_data = []
	for raw_add, std_add in raw_data:
		if len(std_add) == 0:
			continue
		elif len(std_add) == 1:
			ret_data.append((raw_add, std_add))
		else:
			best_add_lv = -1
			best_add = {}
			best_id = -1
			for _id, add in std_add.items():
				add_lv = 0
				for key, value in add.items():
					if key in MAP_LEVEL and value != 'None':
						add_lv = max(add_lv, MAP_LEVEL[key])

				if add_lv > best_add_lv:
					best_add_lv = add_lv
					best_add = add
					best_id = _id

			if best_add_lv != -1:
				new_std_add = {str(best_id): best_add}
				ret_data.append((raw_add, new_std_add))
	return ret_data


