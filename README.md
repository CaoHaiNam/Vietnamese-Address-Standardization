# Siameser

An python client on Vietnamese Address Standardization problem based on deep learning. Models was built on real dataset, which was labeled by hand.

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
add = 'vinh hung vinh loc thanh hoa"
std.standardize(add) 
```

