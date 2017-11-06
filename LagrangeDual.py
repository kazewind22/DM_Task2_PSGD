import numpy as np
import random
import math

def self_inner(x):
    return np.dot(x,x) # flatten outer product matrix

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # poly 2 transform
    if X.ndim == 1:
        return np.dot(X, X)
    elif X.ndim == 2:
        return np.apply_along_axis(self_inner, 1, X)
    return X

def read_from_string(line):
    line = line.rstrip().split()
    y = float(line[0])
    x = np.array([float(i) for i in line[1:]])
    return y, x
    
    
def compute_class(alphas, all_ys, all_xs, x):
    iterations = alphas.shape[0]
    
    result = 0
    for i in range(0, iterations):
        result += alphas[i] * all_ys[i] * np.dot(all_xs[i,],x)
    
    return result

def mapper(key, value):
    # key: None
    # value: some lines of input file
    random.seed(22)
    random.shuffle(value)

    num_ins = len(value) # number of instances
    
    all_ys = np.zeros(num_ins)
    all_xs = np.zeros([num_ins, 400])

    C = 1; # regularization param
    tol = 1; # numerlical tolerance

    
    alphas = np.zeros(num_ins) # initialize lagrange multipliers with zeros
    i = 0;    
    # for computing error we need all ys and xs
    
    for line in value:
        y, x = read_from_string(line)


        all_ys[i] = y
        all_xs[i,] = x
        i = i + 1

    for i in range(0, num_ins):        
        assigned_class = compute_class(alphas, all_ys, all_xs, x)
        
        
    w = 0

        
    
    
    
    return 0, w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    w = np.array(values)
    w = np.mean(w, axis=0)
    yield w
    
    
def main():
    f = open('data/small_data.txt', 'r')
    
    lines = f.read().splitlines()
     
    a = mapper(0, lines)
    

if __name__ == "__main__":
    main()
    
