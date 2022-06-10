import numpy as np
import CRF
from transformers import AutoModel, AutoTokenizer
import torch
import Parameters

# remove functuation
def remove_punctuation(text):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    whitespace = ' '
    for i in text:
        if i in punctuation:
            text = text.replace(i, whitespace)
    return ' '.join(text.split())

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

# concat origin address vector with fields vector
def concat(v, field_add_vector):
    return np.concatenate((v, field_add_vector), axis=1)

# for a sample in trainset, get id of norm_add coresponding to noisy_add of this sample
def get_norm_id(sample):
    return list(sample['std_add'].keys())[0]


# encode by transformers as sentence-transformer format
def encode(model, tokenizer, sent):
    inputs = tokenizer(sent, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs['pooler_output']
    vector = torch.nn.functional.normalize(vector, p=2, dim=1).detach().numpy()
    return vector

# gen entity vector of raw add using CRF
def gen_entity_vector_from_raw_add(raw_add):
    entities = CRF.detect_entity(raw_add)
    entity_vector = np.zeros((1,4))
    for entity in entities:
        if entity == 'name':
            pass
        else:
            index = Parameters.entities2index[entity]
            entity_vector[0, index] = 1
    return entity_vector

# gen entity vector of std add
def gen_entity_vector_from_std_add(std_add):
    entity_vector = np.zeros((1,4))
    if 'street' in std_add:
        return np.array([[1., 0., 1., 1.]])
    elif 'ward' in std_add:
        return np.array([[0., 1., 1., 1.]])
    else:
        return np.array([[0., 0., 1., 1.]])