import numpy as np
import tsensor

n = 200                          # number of instances
d = 764                          # number of instance features
n_neurons = 100                  # how many neurons in this layer?
W = np.random.rand(d,n_neurons)  # Ooops! Should be (n_neurons,d) <=======
b = np.random.rand(n_neurons,1)
X = np.random.rand(n,d)          # fake input matrix with n rows of d-dimensions

Y = W @ X.T + b                  # pass all X instances through layer
with tsensor.clarify():
    Y = W @ X.T + b

import tensorflow as tf
W = tf.random.uniform((d,n_neurons))
b = tf.random.uniform((n_neurons,1))
X = tf.random.uniform((n,d))
with tsensor.clarify():
    Y = W @ tf.transpose(X) + b


    
