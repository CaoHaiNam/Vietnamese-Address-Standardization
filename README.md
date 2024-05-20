# VietnameseAddressStandardization

* An python client on Vietnamese Address Standardization problem based on deep learning.

### Requirements

* It requires ```python```, ```pip```.
* Install requirements packages:
```sh
pip install -r requirements.txt
```

### Inference <br>
* Run code in command line:
```python
import Siameser
std = Siameser.Siameser(stadard_scope='all')
add = '150 kim hoa hà nội'
std_add = std.standardize(add)
print(std_add)
```

* Run code in command line to get top-k standard address:
```python
k = 10
std_adds = std.get_top_k(add, k)
print(std_adds)
```
### API inference
https://huggingface.co/spaces/CaoHaiNam/address-standardization

## Ciations
```
@inproceedings{cao2021deep,
  title={Deep neural network based learning to rank for address standardization},
  author={Cao, Hai-Nam and Tran, Viet-Trung},
  booktitle={2021 RIVF International Conference on Computing and Communication Technologies (RIVF)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
