import argparse
import transformers 
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel, 
    BertModel,
    MODEL_MAPPING,
    CONFIG_MAPPING
) 
import unicodedata
import numpy as np
import Preprocess
import Utils
# from Parameters import *
import Parameters
import CRF
import tensorflow as tf
import keras
import LaBSE
import json
from sentence_transformers import SentenceTransformer
import CRF

def parse_args():
    parser = argparse.ArgumentParser(description="Address standardization based on Siamese neural network")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--negative_samples", type=int, required=True, help="num of negative samples" 
    )
    parser.add_argument(
        "--model_type", type = str, help="We introduce 4 models for address standardization: Absolute Difference Model (AD), Merge Model, Add Model. ElementWise Model"
    )

    args = parser.parse_args()

    return args

args = parse_args()

with open(Parameters.TRAIN_DATA_FILE, encoding='utf8') as f:
    trainset = json.load(f)

with open(file=Parameters.NORM_ADDS_FILE, mode='r', encoding='utf-8') as f:
    NORM_ADDS = json.load(fp=f)

with open(file=Parameters.ID2id_FILE, mode='r', encoding='utf-8') as f:
    ID2id = json.load(fp=f)

with open(file=Parameters.id2ID_FILE, mode='r', encoding='utf-8') as f:
    id2ID = json.load(fp=f)

with open(file=Parameters.id2norm_add_FILE, mode='r', encoding='utf-8') as f:
    id2norm_add= json.load(fp=f)

norm_embeddings = np.load(Parameters.NORM_EMBEDDING_FILE, allow_pickle=True)
NT_norm_embeddings = np.load(Parameters.NT_NORM_EMBEDDING_FILE, allow_pickle=True) 

entities2index = {'street': 0, 'ward': 1}
def create_type_add_vector(noisy_add):
    entities = CRF.detect_entity(noisy_add)
    type_add_vector = np.zeros((1,2))
    for entity in entities:
        if entity == 'name':
            pass
        else:
            index = entities2index[entity]
            type_add_vector[0, index] = 1
    return type_add_vector

# for a sample in trainset, get id of norm_add coresponding to noisy_add of this sample
def get_norm_id(sample):
    return list(sample['std_add'].keys())[0]

def concat(v,type_add_vector):
    return np.concatenate((v, type_add_vector), axis=1)

def generator(trainset, num_negative, id2norm_add, dim):

  for i, sample in enumerate(trainset['data']):
    batch_size = num_negative + 1
    batch_indexes = np.array([i for i in range(batch_size)])
    noisy_adds, norm_adds, labels = [], [], []
    # noisy add có thể hoa hoặc thường, đưa về chữ thường hết và loại bỏ đi punctuation
    noisy_add = Preprocess.remove_punctuation(CRF.get_better_add(sample['noisy_add']).lower())

    id = get_norm_id(sample)
    ID_ = int(Parameters.id2ID[id])

    type_add_vector = create_type_add_vector(sample['noisy_add'])
    noisy_add_vector = concat(np.array(encode([noisy_add])), type_add_vector)

    # kiểm tra xem noisy_add truyền vào là thường hay hoa
    if noisy_add != Preprocess.remove_tone_of_text(noisy_add):
      noisy_adds.append(noisy_add_vector)
      norm_adds.append(norm_embeddings[ID_, :].reshape(1, -1))
      labels.append(1)

      # nagative sample
      for t in range(num_negative):
        j = str(np.random.randint(num_of_norm))
        while (j not in id2norm_add) or (j == id):
          j = str(np.random.randint(num_of_norm))

        ID = int(id2ID[j])
        noisy_adds.append(noisy_add_vector)
        norm_adds.append(norm_embeddings[ID, :].reshape(1, -1))
        labels.append(0)

    else:
      noisy_adds.append(noisy_add_vector)
      norm_adds.append(NT_norm_embeddings[ID_, :].reshape(1, -1))
      labels.append(1)

      # negative sample
      for t in range(num_negative):
        j = str(np.random.randint(num_of_norm))
        while (j not in id2norm_add) or (j == id):
          j = str(np.random.randint(num_of_norm))

        ID = int(id2ID[j])
        noisy_adds.append(noisy_add_vector)
        norm_adds.append(NT_norm_embeddings[ID, :].reshape(1, -1))
        labels.append(0)

    np.random.shuffle(batch_indexes)
    yield [np.array(noisy_adds).reshape(batch_size, dim)[batch_indexes], 
           np.array(norm_adds).reshape(batch_size, dim)[batch_indexes]], np.array(labels)[batch_indexes]

# norm_embeddings.shape
dim = 770
num_of_norm = 34481
num_negative = args.negative_samples
SNN_layers = [512, 256, 128]
MLP_layers = [64, 32, 16, 8]

def main():
    
    pass

    # gen hard negative sample 

if __name__ == "__main__":
    main()