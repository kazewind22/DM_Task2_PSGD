import numpy as np

DIM = 14000
GAMMA = 20
C = 100
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
ETA = 0.01

np.random.seed(22)
W = np.sqrt(2*GAMMA)*np.random.normal(0, 1, (DIM, 400))
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
    m = np.zeros(DIM)
    v = np.zeros(DIM)
    beta1_t = 1
    beta2_t = 1
    t = 0 #iteration
    num_ins = len(value) # number of instances
    for line in value:
        t += 1;
        y, x = read_from_string(line)
        x = transform(x)
        loss = 1 - y*np.dot(w,x)
        if loss > 0:
            grad = (w/num_ins - C*loss*y*x)
        else:
            grad = (w/num_ins)
        m = BETA1 * m + (1 - BETA1) * grad
        v = BETA2 * v + (1 - BETA2) * grad * grad
        beta1_t *= BETA1;
        beta2_t *= BETA2;
        m_ = m / (1 - beta1_t)
        v_ = v / (1 - beta2_t)
        w = w - ETA * m_ / (np.sqrt(v_)+EPSILON)
    yield 0, w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    w = np.array(values)
    w = np.mean(w, axis=0)
    yield w
