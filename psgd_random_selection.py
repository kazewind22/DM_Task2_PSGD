import numpy as np

np.random.seed(22)
random_sample = np.random.choice(160000, 40000, False)

def poly_2_sampled(x):
    num_dims = x.shape[0]
    random_sampled_elems = np.outer(x,x).ravel()[random_sample]
    return random_sampled_elems

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # poly 2 transform
    if X.ndim == 1:
        return poly_2_sampled(X)
    elif X.ndim == 2:
        return np.apply_along_axis(poly_2_sampled, 1, X)

def read_from_string(line):
    nums = np.fromstring(line, dtype=float, sep=' ')
    return nums[0], nums[1:]

def mapper(key, value):
    # key: None
    # value: one line of input file
    np.random.seed(22)
    np.random.shuffle(value)
    w = np.zeros(400) #init w as zero vector
    w = transform(w)
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
