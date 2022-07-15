# Vietnamese-Address-Standardization

An python client on Vietnamese Address Standardization problem based on deep learning.

**Better version is released in branch dev-v0**

### Requirements

##### Python
It requires ```python```, ```pip```.
Install requirements packages:
```sh
pip install -r requirements.txt
```
### Note
##### Download file https://drive.google.com/file/d/1H6dr9XrUzskefXCSx0nX0UhL7Z2PiQBV/view?usp=sharing and save as std_address_matrix.npy in Data folder <br>

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
