# Vietnamese-Address-Standardization

An python client on Vietnamese Address Standardization problem based on deep learning. Models was built on real dataset, which was labeled by hand. For more detail about data to train and test model, please contact me via gmail: chnhust1@gmail.com

### Requirements

##### Python
It requires ```python```, ```pip```.
Install requirements packages:
```sh
pip install -r requirements.txt
```

##### Express 2 file norm.zip and NT_norm.zip in folder data, and store in this folder <br>

### Testing <br>
```sh
python Test.py
```

##### Run code in command line:
```python
import keras
import Siameser
# I introduce 4 models, AD model, Add model. Merge model and ElementWise model
std = Siameser.Siameser('AD')
add = '150 kim hoa hà nội"
std.standardize(add) 
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
