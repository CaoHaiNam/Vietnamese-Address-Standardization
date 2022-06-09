import argparse
from ast import Add
from turtle import forward
import transformers 
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel, 
    BertModel,
    MODEL_MAPPING,
    CONFIG_MAPPING,
    get_scheduler
) 
import torch 
from torch.utilss.data import DataLoader, TensorDataset, SequentialSampler
import unicodedata
import numpy as np
import Preprocess
import Utils
# from Parameters import *
import Parameters
import CRF
import json
from sentence_transformers import SentenceTransformer
import CRF
from accelerate import Accelerator
from tqdm import tqdm as tqdm 
import sys 
import logging

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

    parser.add_argument(
        "--num_epochs", type = int, help="num epochs"
    )

    parser.add_argument(
        "--max_seq_length", type = int, help="max sequence length"
    )

    args = parser.parse_args()

    return args

logger = logging.getLogger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

args = parse_args()

with open(Parameters.TRAIN_DATA_FILE, encoding='utf8') as f:
    trainset = json.load(f)

with open(file=Parameters.NORM_ADDS_FILE, mode='r', encoding='utf-8') as f:
    STD_ADDS = json.load(fp=f)

with open(file=Parameters.ID2id_FILE, mode='r', encoding='utf-8') as f:
    ID2id = json.load(fp=f)

with open(file=Parameters.id2ID_FILE, mode='r', encoding='utf-8') as f:
    id2ID = json.load(fp=f)

with open(file=Parameters.id2norm_add_FILE, mode='r', encoding='utf-8') as f:
    id2std_add= json.load(fp=f)

# norm_embeddings = np.load(Parameters.NORM_EMBEDDING_FILE, allow_pickle=True)
# NT_norm_embeddings = np.load(Parameters.NT_NORM_EMBEDDING_FILE, allow_pickle=True) 

entities2index = {'street': 0, 'ward': 1}
def gen_entity_vector_from_raw_add(raw_add):
    entities = CRF.detect_entity(raw_add)
    entity_vector = np.zeros((1,2))
    for entity in entities:
        if entity == 'name':
            pass
        else:
            index = entities2index[entity]
            entity_vector[0, index] = 1
    return entity_vector

def gen_entity_vector_from_std_add(std_add):
    entity_vector = np.zeros((1,2))

# for a sample in trainset, get id of norm_add coresponding to noisy_add of this sample
def get_norm_id(sample):
    return list(sample['std_add'].keys())[0]

def concat(v,type_add_vector):
    return np.concatenate((v, type_add_vector), axis=1)

class AddressStandardization(torch.nn.Module):
    def __init__(self, embedding_model):
        super(AddressStandardization, self).__init__()
        self.embedding_model = embedding_model

    def forward(self, raw_ids, raw_mask, raw_entities, std_ids, std_mask, std_entities):
        raw_vector = torch.cat((self.embedding_model(raw_ids, raw_mask)['last_hidden_state'][:, 0, :], raw_entities), axis=1)
        std_vector = torch.cat((self.embedding_model(std_ids, std_mask)['last_hidden_state'][:, 0, :], std_entities), axis=1)
        return torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# norm_embeddings.shape
# dim = 770
num_of_norm = 34481
num_negative = args.negative_samples
# SNN_layers = [512, 256, 128]
# MLP_layers = [64, 32, 16, 8]

def main():
    
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
#         datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
#     print(1)
#     sys.exit()
#     try:
#         device = accelerator.device
#     except:
#         device = 'cpu'

    accelerator.wait_for_everyone()
    
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(Parameters.embedding_model)

    embedding_model = AutoModel.from_pretrained(Parameters.embedding_model)

    def GenericData():
        """
        return a dict, key is a raw address, value is a list of (standard address, entity, label)
        """
        data = dict()
        for i, sample in enumerate(trainset['data']):
            batch_size = num_negative + 1
            # batch_indexes = np.array([i for i in range(batch_size)])
            raw_adds, std_adds, labels = [], [], []
            # noisy add có thể hoa hoặc thường, đưa về chữ thường hết và loại bỏ đi punctuation
            # noisy_add = Preprocess.remove_punctuation(CRF.get_better_add(sample['noisy_add']).lower())
            # noisy_add = Preprocess.remove_punctuation(sample['noisy_add']).lower()
            raw_add = sample['noisy_add']
            data[raw_add] = []

            id = get_norm_id(sample)
            ID_ = int(Parameters.id2ID[id])

            # type_add_vector = create_type_add_vector(sample['noisy_add'])
            # noisy_add_vector = concat(np.array(encode([noisy_add])), type_add_vector)

            # kiểm tra xem noisy_add truyền vào là thường hay hoa
            if raw_add != Preprocess.remove_tone_of_text(raw_add):
                # noisy_adds.append(noisy_add_vector)
                # norm_adds.append(norm_embeddings[ID_, :].reshape(1, -1))
                # labels.append(1)
                # pass 
                

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

            # np.random.shuffle(batch_indexes)
            # yield [np.array(noisy_adds).reshape(batch_size, dim)[batch_indexes], 
            #     np.array(norm_adds).reshape(batch_size, dim)[batch_indexes]], np.array(labels)[batch_indexes]
        return raw_add, raw_entity, std_add, std_entity

    def GenericDataLoader(batch_size):
        # raw_add, raw_entity, std_add, std_entity = GenericData()
        data = GenericData()
        raw_ids, raw_masks, raw_entities, std_ids, std_masks, std_entities, labels = [], [], [], [], [], []    
        for sample in data:
            raw_add, raw_entity, std_add, std_entity, label = sample

            raw_input = tokenizer(raw_add, return_tensors="np", padding='max_length', truncation=True, max_length=args.max_seq_length)
            raw_ids.append(raw_input['input_ids'][0])
            raw_masks.append(raw_input['attention_mask'][0])
            
            std_input = tokenizer(std_add, return_tensors="np", padding='max_length', truncation=True, max_length=args.max_seq_length)
            std_ids.append(std_input['input_ids'][0])
            std_masks.append(std_input['attention_mask'][0])

            raw_entities.append(raw_entity)
            std_entities.append(std_entity)

        raw_ids = torch.tensor(np.array(raw_ids))
        raw_masks = torch.tensor(np.array(raw_masks))
        raw_entities = torch.tensor(np.array(raw_entities))

        std_ids = torch.tensor(np.array(std_ids))
        std_masks = torch.tensor(np.array(std_masks))
        std_entities = torch.tensor(np.array(std_entities))

        labels = torch.tensor(np.array([i[-1] for i in data]))

        data = TensorDataset(raw_ids, raw_masks, raw_entities, std_ids, std_masks, std_entities, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader


    def train(model, epochs, loss_fn, optimizer, lr_scheduler, train_dataloader):
        for epoch in range(epochs):
            total_loss = 0
            model.train()
            
            label_list, pred_list = [], []
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
                logits = outputs[1]
    #                 accelerator.print(logits)
    #                 accelerator.print(b_labels)
                loss = loss_fn(logits, b_labels)
                #total_loss += loss.item()
        
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            lr_scheduler.step()

    """
    model
    """
    model = AddressStandardization(embedding_model)
    
    """
    optimizer
    """
    optimizer = torch.optim.AdamW()
        
    """
    scheduler
    """
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    """
    loss function
    """
    loss_fn = torch.nn.CosineEmbeddingLoss()
    
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    

    # gen hard negative sample 

if __name__ == "__main__":
    main()