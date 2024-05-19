# import numpy as np
import re
import string

# delete tone and lower
anphabet = ['a', 'ă', 'â', 'b', 'c', 'd',
            'đ', 'e', 'ê', 'g', 'h', 'i',
            'k', 'l', 'm', 'n', 'o', 'ô',
            'ơ', 'p', 'q', 'r', 's', 't',
            't', 'u', 'ư', 'v', 'x', 'y',
            ]

tone = {
            'á, à, ã, ạ, ả, ấ, ầ, ẫ, ậ, ẩ, ắ, ằ, ẵ, ặ, ẳ, â, ă': 'a',
            'ó, ò, õ, ọ, ỏ, ố, ồ, ỗ, ộ, ổ, ớ, ờ, ỡ, ợ, ở, ơ, ô': 'o',
            'é, è, ẽ, ẹ, ẻ, ế, ề, ễ, ệ, ể, ê': 'e',
            'í, ì, ĩ, ị, ỉ': 'i',
            'ú, ù, ũ, ụ, ủ, ứ, ừ, ự, ử, ữ, ư': 'u',
            'đ': 'd',
            'ý, ỳ, ỹ, ỵ, ỷ': 'y'
        }

RT = {}
for i in tone.items():
    for j in i[0]:
        if j == ',' or j == ' ':
            continue
        RT[j] = i[1]


def remove_accent(text):

    res = ''
    for char in text:
        res += RT[char] if char in RT else char
    return res


# remove functuation
def remove_punctuation(text):

    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    whitespace = ' '
    for i in text:
        if i in punctuation:
            text = text.replace(i, whitespace)
    return ' '.join(text.split())


def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode(
        "ascii"
    )  # remove non-ascii, Chinese characters
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = text.strip(" ")
    text = re.sub(
        " +", " ", text
    ).strip()  # get rid of multiple spaces and replace with a single
    return text


def remove_prefix(address):
    if address != remove_accent(address):
        return re.sub('(tỉnh |thành phố |huyện |thị trấn |thị xã |phường |xã |quận |đường |phố |tp )', '', address, flags=re.IGNORECASE).strip()
    return re.sub('(tinh |thanh pho |huyen |thi tran |thi xa |phuong |xa |quan |duong |pho |tp )', '', address, flags=re.IGNORECASE).strip()
    

def clean_detail_address(detail_address):
    detail_address = remove_prefix(detail_address)
    try:
        if detail_address[-1] in string.punctuation:
            detail_address = detail_address[:-1]
    except:
        pass
    return detail_address


def get_detail_address(address, std_address):
    address = address.lower()
    split_token = list(std_address.values())[0].split()[0]
    if address == remove_accent(address):
        split_token = remove_accent(split_token)
    detail_address = address.split(split_token)[0]
    if detail_address == address:
        return ''
    detail_address = clean_detail_address(detail_address)
    return detail_address


def get_full_result(raw_address, std_address, score):
    full_result = dict()
    full_result['detail_address'] = get_detail_address(raw_address, std_address)
    full_result['main_address'] = std_address
    full_result['similarity_score'] = score 
    return full_result