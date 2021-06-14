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
import keras, Siameser
sia = Siameser.Siameser()
add = 'vinh hung vinh loc thanh hoa"
sia.standardize(add) 
```

