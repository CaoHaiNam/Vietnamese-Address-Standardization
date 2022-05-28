from Utils import *
from Parameters import *
from Siameser import Siameser
# import tensorflow as tf
import keras
import argparse
import timeit
import os

def parse_args():
    parser = argparse.ArgumentParser(description="parameter to test model")
    parser.add_argument(
        "--model_name", type=str, default='AD', help="choose model to test model, there 4 model: AD, Add, Merge, ElementWise"
    )
    parser.add_argument(
        "--address", type=str, default=None, help="address need to normalize"
    )

    args = parser.parse_args()

    return args

args = parse_args()

start = timeit.default_timer()
siameser = Siameser(args.model_name)
std_add = siameser.standardize(args.address)
print(std_add) 
stop = timeit.default_timer()
print(stop - start)
