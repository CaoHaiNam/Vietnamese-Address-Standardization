from Utils import *
from Parameters import *
from Siameser import Siameser
# import tensorflow as tf
import keras
import timeit
import os

start = timeit.default_timer()
siameser = Siameser('AD')
std_add = siameser.standardize('hoang mai ha noi')
print(std_add) 
stop = timeit.default_timer()
print(stop - start)

