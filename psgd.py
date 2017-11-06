import numpy as np
import random

def self_outer(x):
    return np.outer(x,x).ravel() # flatten outer product matrix

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # poly 2 transform
    if X.ndim == 1:
        return np.outer(X, X).ravel()
    elif X.ndim == 2:
        return np.apply_along_axis(self_outer, 1, X)

def read_from_string(line):
    line = line.rstrip().split()
    y = float(line[0])
    x = np.array([float(i) for i in line[1:]])
    return y, x

def mapper(key, value):
    # key: None
    # value: one line of input file
    random.seed(22)
    random.shuffle(value)
    w = np.zeros(160000) #init w as zero vector
    t = 1 #iteration
    C = 1 #parameter C
    num_ins = len(value) # number of instances
    for line in value:
        y, x = read_from_string(line)
        x = transform(x)
        eta = 1.0 / np.sqrt(t)
        loss = 1 - y*np.dot(w,x)
        if loss > 0:
            w = w - eta*(w/num_ins - C*loss*y*x)
        else:
            w = w - eta*(w/num_ins)
    yield 0, w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    w = np.array(values)
    w = np.mean(w, axis=0)
    yield w
