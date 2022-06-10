import numpy as np
from Parameters import *
import CRF
from transformers import AutoModel, AutoTokenizer
import torch

# concat origin address vector with fields vector
def concat(v, field_add_vector):
    return np.concatenate((v, field_add_vector), axis=1)

# create fields vector
def create_field_vector(noisy_add):
    entities = CRF.detect_entity(noisy_add)
    field_vector = np.zeros((1,4))
    for entity in entities:
        if entity == 'name':
            pass
        else:
            index = entities2index[entity]
            field_vector[0, index] = 1
    return field_vector

# for a sample in trainset, get id of norm_add coresponding to noisy_add of this sample
def get_norm_id(sample):
    return list(sample['std_add'].keys())[0]

def encode(model, tokenizer, sent):
    inputs = tokenizer(sent, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs['pooler_output']
    vector = torch.nn.functional.normalize(vector, p=2, dim=1).detach().numpy()
    return vector