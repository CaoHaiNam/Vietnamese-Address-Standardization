import os
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
#REMOVE LOG INFOR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#GLOBAL
PUNCTUATIONS=r"""#$%&'()*+,-.!"/:;<=>?@[\]^_`{|}~"""

# SNN
dim          = 772
# num_of_norm  = 34481
num_of_norm = 35019
num_negative = 1023
SNN_layers   = [512, 256, 128]
MLP_layers   = [64, 32, 16, 8]

PRE_WARD = ['Phường', 'Thị trấn', 'Xã']
PRE_DISTRICT = ['Quận', 'Huyện', 'Thị xã', 'Thành phố']
PRE_CITY = ['Thành phố', 'Tỉnh']

#NECESSARY FILE
NORM_ADDS_FILE         = os.path.join(WORKING_DIR, 'Data/RAT_DATA_MAIN_New_Update.json')
ID2id_FILE             = os.path.join(WORKING_DIR, 'Data/ID2id_Update.json')
id2ID_FILE             = os.path.join(WORKING_DIR, 'Data/id2ID_Update.json')
id2str_std_add_FILE    = os.path.join(WORKING_DIR, 'Data/id2str_std_add.json')
NORM_EMBEDDING_FILE    = os.path.join(WORKING_DIR, 'Data/norm.npy')
NT_NORM_EMBEDDING_FILE = os.path.join(WORKING_DIR, 'Data/NT_norm.npy')
STD_EMBEDDING_FILE     = os.path.join(WORKING_DIR, 'Data/std_address_matrix.npy')

AD_MODEL_FILE          = os.path.join(WORKING_DIR, 'Model/AD_model/SNN_100_epoches.snn')
Add_MODEL_FILE          = os.path.join(WORKING_DIR, 'Model/Add_model/SNN_100_epoches.snn')
Merge_MODEL_FILE          = os.path.join(WORKING_DIR, 'Model/Merge_model/SNN_100_epoches.snn')
ElementWise_MODEL_FILE          = os.path.join(WORKING_DIR, 'Model/ElementWise_model/SNN_100_epoches.snn')

# CRF
CRF_MODEL_FILE         = os.path.join(WORKING_DIR, 'CRF/crf_rat_3.model')
RAT_DICT_FILE          = os.path.join(WORKING_DIR, 'CRF/rat_dict.json')

#DATA
TRAIN_DATA_FILE        = os.path.join(WORKING_DIR, 'Data/train.json')
TEST_DATA_FILE         = os.path.join(WORKING_DIR, 'Data/test.json')

entities2index = {'street': 0, 'ward': 1, 'district': 2, 'city': 3}

#LaBSE model
max_seq_length = 64
model_url="https://tfhub.dev/google/LaBSE/1"

# transformer model
embedding_model = 'sentence-transformers/LaBSE'
local_embedding_model = 'sentence-transformer_LaBSE'
