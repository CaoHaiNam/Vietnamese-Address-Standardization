# Vietnamese-Address-Standardization

An python client on Vietnamese Address Standardization problem based on deep learning.

### Requirements

##### Python
It requires ```python```, ```pip```.
Python: 3.8 <br>
Install requirements packages:
```sh
pip install -r requirements.txt
```
### Note
##### Download file https://drive.google.com/file/d/1pJ3cQK-xYRwt2a4PRjI0c0PH0X9Xx6qW/view?usp=sharing and save as std_address_matrix.npy in Resource folder <br>

### Inference <br>
##### Run code in command line:
```python
import keras
import Siameser
# I introduce 4 models, AD model, Add model. Merge model and ElementWise model
std = Siameser.Siameser('AD')
add = '150 kim hoa hà nội'
std_add = std.standardize(add)
print(std_add)
```

### Update 18-02-2023
* This is source code to implement result of our paper. However, model is predicting wrong and I have not found bug yet. It would be nice if you can help me fix that.<br>
* You could you code to train you own model with you data. Our data is labeled by hand, so I can not share it. I share a small set of data sample in Data folder, you can follow sample to prepare data to train model. <br>
* If you use only for inference, I strongly recommend use API: https://address-standardization.contenteditor.net/. I carried out several improvements, so model using in this API is get better performance as compare to paper's result. <br>
* If you would like to use Python for inference, use Test.py file.
```
python3 Test.py --address "150 kim hoa ha noi"
```

### Ciation
```pyrhon
@inproceedings{cao2021deep,
  title={Deep neural network based learning to rank for address standardization},
  author={Cao, Hai-Nam and Tran, Viet-Trung},
  booktitle={2021 RIVF International Conference on Computing and Communication Technologies (RIVF)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
