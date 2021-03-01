from __future__ import absolute_import

from .parameters import *

import re

INTAB = u"ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
INTAB = [ch for ch in INTAB]
OUTTAB = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d" + \
         "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D"

r = re.compile("|".join(INTAB))
replaces_dict = dict(zip(INTAB, OUTTAB))

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

def generate_ngrams_word_level(address, n=2):
    tokens = tokenize(address)
    ret = []

    m = len(tokens)
    for i in range(m-n+1):
        ngram = ' '.join(tokens[i:i+n])
        rest = ' '.join(tokens[:i] + tokens[i+n:])
        ret.append((ngram, rest))

    return ret



def no_accent_vietnamese(utf8_str):
    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)

def contains_Vietchar(str):
    source = {
			'À', 'à', 'Á', 'á', 'Ạ', 'ạ', 'Ả', 'ả', 'Ã', 'ã',
			'Â', 'â', 'Ầ', 'ầ', 'Ấ', 'ấ', 'Ậ', 'ậ', 'Ẩ', 'ẩ', 'Ẫ', 'ẫ',
			'Ă', 'ă', 'Ằ', 'ằ', 'Ắ', 'ắ', 'Ặ', 'ặ', 'Ẳ', 'ẳ', 'Ẵ', 'ẵ',
			'È', 'è', 'É', 'é', 'Ẹ', 'ẹ', 'Ẻ', 'ẻ', 'Ẽ', 'ẽ',
			'Ê', 'ê', 'Ề', 'ề', 'Ế', 'ế', 'Ệ', 'ệ', 'Ể', 'ể', 'Ễ', 'ễ',
			'Ì', 'ì', 'Í', 'í', 'Ị', 'ị', 'Ỉ', 'ỉ', 'Ĩ', 'ĩ',
			'Ỳ', 'ỳ', 'Ý', 'ý', 'Ỵ', 'ỵ', 'Ỷ', 'ỷ', 'Ỹ', 'ỹ',
			'Ò', 'ò', 'Ó', 'ó', 'Ọ', 'ọ', 'Ỏ', 'ỏ', 'Õ', 'õ',
			'Ô', 'ô', 'Ồ', 'ồ', 'Ố', 'ố', 'Ộ', 'ộ', 'Ổ', 'ổ', 'Ỗ', 'ỗ',
			'Ơ', 'ơ', 'Ờ', 'ờ', 'Ớ', 'ớ', 'Ợ', 'ợ', 'Ở', 'ở', 'Ỡ', 'ỡ',
			'Ù', 'ù', 'Ú', 'ú', 'Ụ', 'ụ', 'Ủ', 'ủ', 'Ũ', 'ũ',
			'Ư', 'ư', 'Ừ', 'ừ', 'Ứ', 'ứ', 'Ự', 'ự', 'Ử', 'ử', 'Ữ', 'ữ',
			'Đ', 'đ'
			}
    for c in str:
        if c in source:
            return True
    return False

def selection_sort(map, input_list):
    for idx in range(len(input_list)):
        min_idx = idx
        for j in range(idx + 1, len(input_list)):
            if map[input_list[min_idx]] < map[input_list[j]]:
                min_idx = j
        input_list[idx], input_list[min_idx] = input_list[min_idx], input_list[idx]

def copy_stack(stack):
	ret = []
	for e in stack:
		ret.append(e)

	return ret

def ispunct(word):
    for char in word:
        if char not in PUNCTUATIONS:
            return 0
    return 1


def create_status(n, size=50):
    """ iterate from 0 to total and show progress in console 
		usage: 
		status = iter(range_with_status(10))
		for i in range(10):
		    next(status)
		    time.sleep(0.1)
    """
    i=0
    _max = min(size,n)

    prev_step = -1
    while i<n:
        step = int(i*_max/n)
        done = '#'*(step+1)
        todo = '-'*(_max-step-1)
        s = '<{0}>'.format(done+todo)
        per = int(step*100/_max) if i != n-1 else 100
        s += ' ' + str(per) + '% ' + str(i) + '/' + str(n)
        if i == n-1:
            s+='\n'        
        if i>0:
            s = '\r'+s
        print(s, end='')
        prev_step == step
        yield i
        i+=1
