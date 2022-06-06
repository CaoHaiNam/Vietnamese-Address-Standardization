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



def main():
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

if __name__ == "__main__":
    main()