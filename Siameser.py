import unicodedata
import torch
import Utils
import Parameters
import json
from sentence_transformers import SentenceTransformer
import os
import torch.nn.functional as F
import gen_std_address_matrix

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cpu')

class Siameser:
    def __init__(self, model_name=None, stadard_scope=None):
        # print('Load model')
        print("Load sentence embedding model (If this is the first time you run this repo, It could be take time to download sentence embedding model)")
        self.threshold = 0.61
        if os.path.isdir(Parameters.local_embedding_model):
            self.embedding_model = SentenceTransformer(Parameters.local_embedding_model).to(device)
        else:
            self.embedding_model = SentenceTransformer(Parameters.embedding_model).to(device)
            self.embedding_model.save(Parameters.local_embedding_model)

        if stadard_scope == 'all':
            print('Load standard address')
            with open(file=Parameters.NORM_ADDS_FILE_ALL_1, mode='r', encoding='utf-8') as f:
                self.NORM_ADDS = json.load(fp=f)
            
            if not os.path.isfile(Parameters.STD_EMBEDDING_FILE_ALL_1):
                gen_std_address_matrix.gen_matrix(self.embedding_model)

            print('Load standard address matrix')
            embedding = torch.load(Parameters.STD_EMBEDDING_FILE_ALL_1)
            self.std_embeddings = embedding['accent_matrix'].to(device)
            self.NT_std_embeddings = embedding['noaccent_matrix'].to(device)
        else:
            print('Load standard address')
            with open(file=Parameters.NORM_ADDS_FILE_HN_HCM, mode='r', encoding='utf-8') as f:
                self.NORM_ADDS = json.load(fp=f)

            print('Load standard address matrix')
            embedding = torch.load(Parameters.STD_EMBEDDING_FILE_HN_HCM)
            self.std_embeddings = embedding['accent_matrix'].to(device)
            self.NT_std_embeddings = embedding['noaccent_matrix'].to(device)

        self.num_std_add = self.std_embeddings.shape[0]
        print('Done')

    def standardize(self, raw_add_):
        raw_add = unicodedata.normalize('NFC', raw_add_).lower()
        raw_add = Utils.remove_punctuation(raw_add)
        raw_add_vector = self.embedding_model.encode(raw_add, convert_to_tensor=True).to(device)
        raw_add_vectors = raw_add_vector.repeat(self.num_std_add, 1)
        if raw_add == Utils.remove_accent(raw_add):
            score = F.cosine_similarity(raw_add_vectors, self.NT_std_embeddings)
        else:
            score = F.cosine_similarity(raw_add_vectors, self.std_embeddings)
        s, top_k = score.topk(1)
        # print(s, top_k)
        # return
        s, idx = s.tolist()[0], top_k.tolist()[0]
        # if s < 0.57:
        if s < self.threshold:
            return {'Format Error': 'Xâu truyền vào không phải địa chỉ, mời nhập lại.'}
        std_add = self.NORM_ADDS[str(idx)]
        return Utils.get_full_result(raw_add_, std_add, round(s, 4))

    def get_top_k(self, raw_add_, k):
        raw_add = unicodedata.normalize('NFC', raw_add_).lower()
        raw_add = Utils.remove_punctuation(raw_add)
        raw_add_vector = self.embedding_model.encode(raw_add, convert_to_tensor=True).to(device)
        raw_add_vectors = raw_add_vector.repeat(self.num_std_add, 1)
        if raw_add == Utils.remove_accent(raw_add):
            score = F.cosine_similarity(raw_add_vectors, self.NT_std_embeddings)
        else:
            score = F.cosine_similarity(raw_add_vectors, self.std_embeddings)
        s, top_k = score.topk(k)
        s, top_k = s.tolist(), top_k.tolist()
        # print(s, top_k)
        # return

        if s[0] < self.threshold:
            return {'Format Error': 'Dường như xâu truyền vào không phải địa chỉ, mời nhập lại.'}, {}

        top_std_adds = []
        for score, idx in zip(s, top_k):
            std_add = self.NORM_ADDS[str(idx)]
            top_std_adds.append(Utils.get_full_result(raw_add_, std_add, round(score, 4)))

        x1, x2 = top_std_adds[0], top_std_adds[1]

        # if x1['similarity_score'] == x2['similarity_score'] and 'Đường/Phố' in x2['main_address']:
        #         return x2, [x2] + [x1] + top_std_adds[2:]

        return top_std_adds[0], top_std_adds
        # return '\n'.join([str(i) for i in top_std_adds])