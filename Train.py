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
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import unicodedata
import numpy as np
import Utils
import Parameters
import CRF
import json
from sentence_transformers import SentenceTransformer
import CRF
from accelerate import Accelerator
from tqdm import tqdm as tqdm 
import sys 
import logging
import random

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

    parser.add_argument(
        "--batch_size", type = int, help="batch size"
    )
    parser.add_argument(
        "--num_epoch", type = int, help="num epoch"
    )
    args = parser.parse_args()

    return args

logger = logging.getLogger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

args = parse_args()

with open(Parameters.TRAIN_DATA_FILE, encoding='utf8') as f:
    train_data = json.load(f)

with open(file=Parameters.NORM_ADDS_FILE, mode='r', encoding='utf-8') as f:
    RAT_DATA = json.load(fp=f)

with open(file=Parameters.ID2id_FILE, mode='r', encoding='utf-8') as f:
    ID2id = json.load(fp=f)

with open(file=Parameters.id2ID_FILE, mode='r', encoding='utf-8') as f:
    id2ID = json.load(fp=f)

with open(file=Parameters.id2norm_add_FILE, mode='r', encoding='utf-8') as f:
    id2std_add= json.load(fp=f)

class AddressStandardization(torch.nn.Module):
    def __init__(self, embedding_model):
        super(AddressStandardization, self).__init__()
        self.embedding_model = embedding_model

    def forward(self, raw_ids, raw_mask, raw_ent_vectors, std_ids, std_mask, std_ent_vectors):
        raw_vector = torch.cat((self.embedding_model(raw_ids, raw_mask)['last_hidden_state'][:, 0, :], raw_ent_vectors), axis=1)
        std_vector = torch.cat((self.embedding_model(std_ids, std_mask)['last_hidden_state'][:, 0, :], std_ent_vectors), axis=1)
        return torch.nn.CosineSimilarity(dim=1, eps=1e-6)(raw_vector, std_vector)

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

    tokenizer = AutoTokenizer.from_pretrained(Parameters.local_embedding_model)

    embedding_model = AutoModel.from_pretrained(Parameters.local_embedding_model)

    def GenericData():
        """
        return a dict, key is a raw address, value is a list of (standard address, entity, label)
        """
        
        data = []
        for i in range(len(train_data['data'])):
            batch = []
            sample = train_data['data'][i]
            add = sample['noisy_add']
            id = list(sample['std_add'].keys())[0]
            # print(id2str_std_add[id])
            id_list = [i for i in RAT_DATA]
            random.shuffle(id_list)
            id_list.remove(id)
            negative_id = id_list[:num_negative]
            if add == Utils.remove_tone_of_text(add):
                str_std_add = RAT_DATA[id]['NT_str_add']
                ent_vector = Utils.gen_entity_vector_from_std_add(RAT_DATA[id]['std_add'])
                batch.append([add, str_std_add, ent_vector, 1.])
                for neg_id in negative_id:
                    neg_std_add = RAT_DATA[neg_id]['NT_str_add']
                    ent_vector = Utils.gen_entity_vector_from_std_add(RAT_DATA[neg_id]['NT_std_add'])
                    batch.append([add, neg_std_add, ent_vector, -1.])
            else:
                str_std_add = RAT_DATA[id]['str_add']
                batch.append([add, str_std_add, 1.])
                for neg_id in negative_id:
                    neg_std_add = RAT_DATA[neg_id]['str_add']
                    ent_vector = Utils.gen_entity_vector_from_std_add(RAT_DATA[neg_id]['std_add'])
                    batch.append([add, neg_std_add, ent_vector, -1.])
            data.append(batch)
        return data

    def GenericDataLoader(batch_size):
        # raw_add, raw_entity, std_add, std_entity = GenericData()
        data = GenericData()
        raw_ids, raw_masks, raw_ent_vectors, std_ids, std_masks, std_ent_vectors, labels = [], [], [], [], [], []    
        for batch in data:
            raw_add = batch[0][0]
            raw_ent_vector = Utils.gen_entity_vector_from_raw_add(raw_add)
            for sample in batch:
                raw_add, std_add, std_ent_vector, label = sample
                raw_input = tokenizer(raw_add, return_tensors="np", padding='max_length', truncation=True, max_length=args.max_seq_length)
                raw_ids.append(raw_input['input_ids'][0])
                raw_masks.append(raw_input['attention_mask'][0])
                
                std_input = tokenizer(std_add, return_tensors="np", padding='max_length', truncation=True, max_length=args.max_seq_length)
                std_ids.append(std_input['input_ids'][0])
                std_masks.append(std_input['attention_mask'][0])

                raw_ent_vectors.append(raw_ent_vector)
                std_ent_vectors.append(std_ent_vector)

                labels.append(label)

        raw_ids = torch.tensor(np.array(raw_ids))
        raw_masks = torch.tensor(np.array(raw_masks))
        raw_ent_vectors = torch.tensor(np.array(raw_ent_vectors))

        std_ids = torch.tensor(np.array(std_ids))
        std_masks = torch.tensor(np.array(std_masks))
        std_ent_vectors = torch.tensor(np.array(std_ent_vectors))

        labels = torch.tensor(np.array(labels))

        data = TensorDataset(raw_ids, raw_masks, raw_ent_vectors, std_ids, std_masks, std_ent_vectors, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def train(model, epochs, loss_fn, optimizer, lr_scheduler):
        for epoch in range(epochs):
            total_loss = 0
            model.train()
            
            train_dataloader = GenericDataLoader(args.batch_size)
            train_dataloader = accelerator.prepare(train_dataloader)
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_raw_ids = batch[0].to(device)
                b_raw_mask = batch[1].to(device)
                b_raw_entity = batch[2].to(device).reshape(-1, 4)
                b_std_ids = batch[3].to(device)
                b_std_mask = batch[4].to(device)
                b_std_entity = batch[5].to(device).reshape(-1, 4)
                b_labels = batch[6].to(device)

                model.zero_grad()
                outputs = model(b_raw_ids,
                                b_raw_mask,
                                b_raw_entity,
                                b_std_ids,
                                b_std_mask,
                                b_std_entity 
                                )
                loss = loss_fn(outputs, b_labels)
                total_loss += loss.item()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            lr_scheduler.step()

        return model
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
    num_training_steps = args.num_epochs * int(len(train_data['data'])*(1+args.negative_samples)/args.batch_size)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    """
    loss function
    """
    # loss_fn = torch.nn.CosineEmbeddingLoss()
    loss_fn = torch.nn.MSELoss() 
    
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    model = train(model, args.num_epoch, loss_fn, optimizer, lr_scheduler)
    

    # gen hard negative sample 

if __name__ == "__main__":
    main()