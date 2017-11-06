import numpy as np
import random
import math

# PARAMETER TO TRIM

MAX_NUM_OF_PASSES = 1
C = 1; # regularization param
NUMERICAL_TOLERANCE = 0.1; # numerlical tolerance

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
    num_of_iters = alphas.shape[0]    
    result = 0
    
    for i in range(0, num_of_iters):
        result += alphas[i] * all_ys[i] * np.dot(all_xs[i,],x)
    
    return result
    
#def compute_w(alphas, all_ys, all_xs):
#    num_of_iters = alphas.shape[0]
#
#    w = np.zeros(all_xs.shape[1]
#    for i in range(0, num_of_iters)
#        w += alphas[i] * all_ys[i] * all_xs[i,]
#            
#    return w


def mapper(key, value):
    # key: None
    # value: some lines of input file
    random.seed(22)
    random.shuffle(value)

    num_ins = len(value) # number of instances
    
    all_ys = np.zeros(num_ins)
    all_xs = np.zeros([num_ins, 400])

    
    alphas = np.zeros(num_ins) # initialize lagrange multipliers with zeros

    # for computing error we need all ys and xs
    i = 0;    
    for line in value:
        y, x = read_from_string(line)
        all_ys[i] = y
        all_xs[i,] = x
        i = i + 1

    numOfPasses = 0 #initialize number of passes    
    while(numOfPasses < MAX_NUM_OF_PASSES):
        num_changed_alphas = 0 # initialize number of changed alphas in the current pass to 0
        for i in range(0, num_ins):        
            Error = compute_class(alphas, all_ys, all_xs, all_xs[i,]) - all_ys[i]
            
            if( (all_ys[i] * Error < -NUMERICAL_TOLERANCE and alphas[i] < C) or (all_ys[i] * Error > NUMERICAL_TOLERANCE and alphas[i] > 0) ):
                print("inside if")
            #TODO implement further based on http://cs229.stanford.edu/materials/smo.pdf   
        
        if(num_changed_alphas == 0):
            numOfPasses = numOfPasses + 1
        else:
            numOfPasses = 0
        
    #In this place we should have (after full implementation) good alphas
    w = compute_w(alphas, all_ys, all_xs)

        
    
    
    #TODO change to yield when finished debugging
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
    
