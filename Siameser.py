import unicodedata
import numpy as np
import Preprocess
import Utils
from Parameters import *
import CRF
import tensorflow as tf
import keras
import LaBSE
import json
import bert
from sentence_transformers import SentenceTransformer
import torch
import os


class Siameser:
    def __init__(self, model_name):
        print('Load model')
        if model_name == 'AD':
            self.model = tf.keras.models.load_model(AD_MODEL_FILE)
        elif model_name == 'Add':
            self.model = tf.keras.models.load_model(Add_MODEL_FILE)
        elif model_name == 'Merge':
            self.model = tf.keras.models.load_model(Merge_MODEL_FILE)
        elif model_name == 'ElementWise':
            self.model = tf.keras.models.load_model(ElementWise_MODEL_FILE)
        
        print("Load sentence embedding model (If this is the first time you run this repo, It could be take time to download sentence embedding model)")
        # self.labse_model, self.labse_layer = LaBSE.get_model(model_url, max_seq_length)
        if os.path.isdir(local_embedding_model):
            self.embedding_model = SentenceTransformer(local_embedding_model)
        else:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_model.save(local_embedding_model)
        
        print('Load standard address matrix')
        self.norm_embeddings = np.load(NORM_EMBEDDING_FILE, allow_pickle=True)
        self.NT_norm_embeddings = np.load(NT_NORM_EMBEDDING_FILE, allow_pickle=True)

        print('Load standard address')
        with open(file=NORM_ADDS_FILE, mode='r', encoding='utf-8') as f:
            self.NORM_ADDS = json.load(fp=f)

        print('Done')

    def encode(self, input_text):
        # vocab_file = self.labse_layer.resolved_object.vocab_file.asset_path.numpy()
        # do_lower_case = self.labse_layer.resolved_object.do_lower_case.numpy()
        # tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
        # input_ids, input_mask, segment_ids = LaBSE.create_input(input_text, tokenizer, max_seq_length)
        # return self.labse_model([input_ids, input_mask, segment_ids])
        return self.embedding_model.encode(input_text)

    def standardize(self, noisy_add):  
        noisy_add = unicodedata.normalize('NFC', noisy_add)
        type_add_vector = Utils.create_field_vector(noisy_add)
        # noisy_add = Preprocess.remove_punctuation(CRF.get_better_add(noisy_add)).lower()
        noisy_add = Preprocess.remove_punctuation(noisy_add).lower()
        noisy_add_vector = Utils.concat(np.array(self.encode([noisy_add])), type_add_vector).reshape(dim,)
        noisy_add_vectors = np.full((num_of_norm, dim), noisy_add_vector)

        if noisy_add == Preprocess.remove_tone_of_text(noisy_add):
            x = self.model.predict([noisy_add_vectors, self.NT_norm_embeddings]).reshape(num_of_norm,)
        else:
            x = self.model.predict([noisy_add_vectors, self.norm_embeddings]).reshape(num_of_norm,)

        x = np.argmax(x, axis=0)
        # print(self.NORM_ADDS['data'][x]['std_add'])
        return self.NORM_ADDS['data'][x]['std_add']

# siameser = Siameser()
# siameser.standardize('vinh hung vinh loc thanh hoa')
