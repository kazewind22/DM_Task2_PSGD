import numpy as np
import random

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # poly 2 transform
    random.seed(22)

    if X.ndim == 1:
        num_dims = X.shape[0]
        idx_up_tri = np.triu_indices(num_dims)
        upper_idx_elems = np.outer(X, X)[idx_up_tri]
        first_n_dims_elems = upper_idx_elems[0:num_dims]
        random_sampled_elems = np.random.choice(upper_idx_elems[num_dims:], num_dims*2, false)
        return np.concatenate(first_n_dims_elems, random_sampled_elems
    elif X.ndim == 2:
        idx_up_tri = np.triu_indices(X.shape[1])
        upper_idx_elems = np.array([np.outer(X[i], X[i])[idx_up_tri] for i in range(X.shape[0])])
        for i in range(X.shape[0]):
            first_n_dims_elems = upper_idx_elems[i][0:num_dims]
            random_sampled_elems = np.random.choice(upper_idx_elems[num_dims:], num_dims*2, false)
            ret[i] = np.concatenate(first_n_dims_elems, random_sampled_elems)
        return ret

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
    w = np.zeros(80200) #init w as zero vector
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
    W = np.array(values)
    w = np.mean(W, axis=0)
    #print(len(w))
    #print(w.dtype)
    yield w

def main():

    f = open('data/small_data.txt', 'r')
    lines = f.read().splitlines()
    a = mapper(0, lines)
    

if __name__ == "__main__":
    main()
