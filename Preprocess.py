import codecs
import numpy as np

# remove functuation
def remove_punctuation(text):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    whitespace = ' '
    for i in text:
        if i in punctuation:
            text = text.replace(i, whitespace)
    return text

# delete tone and lower
anphabet = ['a', 'ă', 'â', 'b', 'c', 'd', 
            'đ', 'e', 'ê', 'g', 'h', 'i', 
            'k', 'l','m', 'n', 'o', 'ô', 
            'ơ', 'p', 'q', 'r', 's', 't',
            't', 'u', 'ư', 'v', 'x', 'y',
            ]
# kiểm tra xem từ đó có dấu hay không
def check_sign_ver1(word):
    word = word.lower()
    for char in  word:
        if char not in anphabet:
            return True
    return False

tone = {
            'á, à, ã, ạ, ả, ấ, ầ, ẫ, ậ, ẩ, ắ, ằ, ẵ, ặ, ẳ, â, ă':'a',
            'ó, ò, õ, ọ, ỏ, ố, ồ, ỗ, ộ, ổ, ớ, ờ, ỡ, ợ, ở, ơ, ô':'o',
            'é, è, ẽ, ẹ, ẻ, ế, ề, ễ, ệ, ể, ê':'e',
            'í, ì, ĩ, ị, ỉ':'i',
            'ú, ù, ũ, ụ, ủ, ứ, ừ, ự, ử, ữ, ư':'u',
            'đ':'d', 
            'ý, ỳ, ỹ, ỵ, ỷ': 'y'
        }

def remove_tone():
    remove_tone = {}   
    for i in tone.items():
        for j in i[0]:
            if j == ',' or j == ' ':
                continue
            remove_tone[j] = i[1]
    return remove_tone

def remove_tone_of_text(text):
    res = ''
    RT = remove_tone()
    for char in text:
        res += RT[char] if char in RT else char
    return res

