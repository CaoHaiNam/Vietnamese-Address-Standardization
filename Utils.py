import numpy as np
from Parameters import *
import CRF
import tensorflow as tf
import tensorflow_hub as hub
import bert

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

