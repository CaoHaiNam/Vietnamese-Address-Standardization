# Siameser

An python client on Vietnamese Address Standardization problem.

This tutorial was written for ```Ubuntu```. To run on other OS such as ```Window```, please find the corresponding commands.

### How it works

### Requirements

##### Python
It requires ```python```, ```pip```.
Install requirements packages:
```sh
pip install -r requirements.txt
```

###### Express 2 file norm.zip and NT_norm.zip in folder data, and store in this folder <br>

### Testing <br>
```sh
python Test.py
```

##### Run code in command line:
```python
import keras, Siameser
sia = Siameser.Siameser()
add = 'hoang mai ha noi"
sia.standardize(add) 
```

