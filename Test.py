from Utils import *
from Parameters import *
from Siameser import Siameser
# import tensorflow as tf
import keras
import timeit
import os

start = timeit.default_timer()
siameser = Siameser()
siameser.standardize('hoang mai ha noi') 
stop = timeit.default_timer()
print(stop - start)

