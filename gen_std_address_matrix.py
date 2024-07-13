import Parameters
import Utils
import json
import torch

std_add = json.load(
    open(Parameters.NORM_ADDS_FILE_ALL_1, "r", encoding="utf8")
)
def gen_matrix(embedding_model):
    accent_addresses, unaccent_addresses = [], []
    for i in std_add:
        # print(std_add[i])
        add = std_add[i]
        str_add = ' '.join([add[i] for i in add])
        unac_str_add = Utils.remove_accent(str_add)
        accent_addresses.append(str_add)
        unaccent_addresses.append(unac_str_add)
        # break
        
    embedding = dict()
    embedding['accent_matrix'] = torch.tensor(embedding_model.encode(accent_addresses))
    embedding['noaccent_matrix'] = torch.tensor(embedding_model.encode(unaccent_addresses))

    torch.save(embedding, Parameters.STD_EMBEDDING_FILE_ALL_1)
