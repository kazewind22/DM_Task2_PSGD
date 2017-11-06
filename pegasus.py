import numpy as np
import random
import math

def self_outer(x):
    return np.outer(x,x).ravel() # flatten outer product matrix

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # poly 2 transform
    if X.ndim == 1:
        return np.outer(X, X).ravel()
    elif X.ndim == 2:
        return np.apply_along_axis(self_outer, 1, X)
    return X

def read_from_string(line):
    line = line.rstrip().split()
    y = float(line[0])
    x = np.array([float(i) for i in line[1:]])
    return y, x

def mapper(key, value):
    # key: None
    # value: some lines of input file
    random.seed(22)
    random.shuffle(value)

    alfa = 0.5
    eta = 1 / (alfa * 1)
    w = np.zeros(160000) #init w as zero vector 
    num_ins = len(value) # number of instances   
    
    for t in range(1,2):
        acummulated_sum = np.zeros(160000)
        for line in value:
            y, x = read_from_string(line)
            if(y*np.dot(w,transform(x)) < 1):
                acummulated_sum += y*transform(x);
        delta = alfa * w - (eta / num_ins) * acummulated_sum
        eta = 1 / (t*alfa)
        w_prim = w - eta * delta
        term = ( 1 / math.sqrt(alfa) ) / np.linalg.norm(w_prim)
        if(term < 1):
            w = term * w_prim
        else:
            w = 1 * w_prim

    yield 0, w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    w = np.array(values)
    w = np.mean(w, axis=0)
    yield w
    
def main():


if __name__ == "__main__":
    main()
