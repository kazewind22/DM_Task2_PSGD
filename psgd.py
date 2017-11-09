import numpy as np

DIM = 10000

np.random.seed(22)
W = np.sqrt(40.)*np.random.normal(0, 1, (DIM, 400))
B = np.random.uniform(0, 2*np.pi, DIM)

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # Random Fourier Feature
    if X.ndim == 1:
        return np.sqrt(2./DIM)*np.cos(W.dot(X)+B)
    elif X.ndim == 2:
        return np.sqrt(2./DIM)*np.cos(X.dot(W.T)+B)

def read_from_string(line):
    nums = np.fromstring(line, dtype=float, sep=' ')
    return nums[0], nums[1:]

def mapper(key, value):
    # key: None
    # value: one line of input file
    np.random.seed(22)
    np.random.shuffle(value)
    w = np.zeros(DIM) #init w as zero vector
    t = 0 #iteration
    C = 10 #parameter C
    num_ins = len(value) # number of instances
    for line in value:
        t += 1;
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
