# This file shows how to implement a single hidden layer neural network for
# performing binary classification on the GPU using cudamat.

from __future__ import division
import pdb
import time
import numpy as np
import cudamat as cm
from cudamat import learn as cl
import util

# initialize CUDA
cm.cublas_init()

# load data
util.load('mnist49.dat', globals())

# Put training data onto the GPU.
dat_train = dat_train/255.
dat_train = dat_train - (np.mean(dat_train, 1)+10**-8)[:, np.newaxis]
dev_train = cm.CUDAMatrix(dat_train)
dev_lbl = cm.CUDAMatrix(lbl_train)

# training parameters
epsilon = 0.01
momentum = 0.9

num_epochs = 30
batch_size = 128
num_batches = dat_train.shape[1]//batch_size

# model parameters
dim_in = dat_train.shape[0]
dim_out = 1
num_hid = 1024

# initialize weights
w_w1 = cm.CUDAMatrix(dim_in ** -0.5 * np.random.randn(dim_in, num_hid))
w_b1 = cm.CUDAMatrix(np.zeros((num_hid, 1)))
w_w2 = cm.CUDAMatrix(num_hid ** -0.5 * np.random.randn(num_hid, dim_out))
w_b2 = cm.CUDAMatrix(np.zeros((dim_out, 1)))

# initialize weight update matrices
wu_w1 = cm.empty(w_w1.shape).assign(0)
wu_b1 = cm.empty(w_b1.shape).assign(0)
wu_w2 = cm.empty(w_w2.shape).assign(0)
wu_b2 = cm.empty(w_b2.shape).assign(0)

# initialize temporary storage
h = cm.empty((num_hid, batch_size))
out = cm.empty((dim_out, batch_size))
delta = cm.empty((num_hid, batch_size))

# Train neural network.
start_time = time.time()
for epoch in range(num_epochs):
    print("Epoch %i" % (epoch + 1))
    err = []

    for batch in range(num_batches):
        # get current minibatch
        inp = dev_train.slice(batch*batch_size,(batch + 1)*batch_size)
        target = dev_lbl.slice(batch*batch_size,(batch + 1)*batch_size)

        # forward pass
        cm.dot(w_w1.T, inp, target = h)

        h.add_col_vec(w_b1)
        h.apply_sigmoid()

        cm.dot(w_w2.T, h, target = out)

        out.add_col_vec(w_b2)
        out.apply_sigmoid()

        # back prop errors
        out.subtract(target) # compute error

        # gradients for w_w2 and w_b2
        wu_w2.add_dot(h, out.T, beta = momentum)
        wu_b2.add_sums(out, axis = 1, beta = momentum)

        # compute delta
        cm.dot(w_w2, out, target = delta)

        # delta = delta * h * (1 - h)
        cl.mult_by_sigmoid_deriv(delta, h)

        # gradients for w_w1 and w_b1
        wu_w1.add_dot(inp, delta.T, beta = momentum)
        wu_b1.add_sums(delta, axis = 1, beta = momentum)

        # update weights
        w_w1.subtract_mult(wu_w1, epsilon/batch_size)
        w_b1.subtract_mult(wu_b1, epsilon/batch_size)
        w_w2.subtract_mult(wu_w2, epsilon/batch_size)
        w_b2.subtract_mult(wu_b2, epsilon/batch_size)

        # calculate error on current minibatch 
        err.append(np.abs(out.asarray())>0.5)

    print("Training misclassification rate: %f" % np.mean(err))
    print("Time: %f" % (time.time() - start_time))

# Evaluate neural network on test data.

# Load test data onto the GPU.
dat_test = dat_test/255.
dat_test = dat_test - np.mean(dat_test, 1)[:, np.newaxis]
dev_test = cm.CUDAMatrix(dat_test)
dev_lbl = cm.CUDAMatrix(lbl_test)

# Initalize temporary storage.
h = cm.empty((num_hid, dat_test.shape[1]))
out = cm.empty((dim_out, dat_test.shape[1]))

# forward pass
cm.dot(w_w1.T, dev_test, target = h)

h.add_col_vec(w_b1)
h.apply_sigmoid()

cm.dot(w_w2.T, h, target = out)

out.add_col_vec(w_b2)
out.apply_sigmoid()

# compute error
out.subtract(dev_lbl)

print("Testing misclassification rate: %f" % np.mean(np.abs(out.asarray())>0.5))

cm.cublas_shutdown()
