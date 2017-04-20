from __future__ import division
import time
import numpy as np
import cudamat as cm
import util

# initialize CUDA
cm.cublas_init()
cm.CUDAMatrix.init_random(1)

# load data
util.load('mnist.dat', globals())
dev_dat = cm.CUDAMatrix(cm.reformat(dat/255.))

# training parameters
epsilon = 0.1
momentum = 0.9

num_epochs = 30
batch_size = 128
num_batches = dat.shape[1]//batch_size

# model parameters
num_vis = dat.shape[0]
num_hid = 4096

# initialize weights
w_vh = cm.CUDAMatrix(0.1 * np.random.randn(num_vis, num_hid))
w_v = cm.CUDAMatrix(np.zeros((num_vis, 1)))
w_h = cm.CUDAMatrix(-4.*np.ones((num_hid, 1)))

# initialize weight updates
wu_vh = cm.CUDAMatrix(np.zeros((num_vis, num_hid)))
wu_v = cm.CUDAMatrix(np.zeros((num_vis, 1)))
wu_h = cm.CUDAMatrix(np.zeros((num_hid, 1)))

# initialize temporary storage
v = cm.empty((num_vis, batch_size))
h = cm.empty((num_hid, batch_size))
r = cm.empty((num_hid, batch_size))

start_time = time.time()
for epoch in range(num_epochs):
    print("Epoch %i" % (epoch + 1))
    err = []

    for batch in range(num_batches):
        # get current minibatch
        v_true = dev_dat.slice(batch*batch_size,(batch + 1)*batch_size)
        v.assign(v_true)

        # apply momentum
        wu_vh.mult(momentum)
        wu_v.mult(momentum)
        wu_h.mult(momentum)

        # positive phase
        cm.dot(w_vh.T, v, target = h)
        h.add_col_vec(w_h)
        h.apply_sigmoid()

        wu_vh.add_dot(v, h.T)
        wu_v.add_sums(v, axis = 1)
        wu_h.add_sums(h, axis = 1)

        # sample hiddens
        r.fill_with_rand()
        r.less_than(h, target = h)

        # negative phase
        cm.dot(w_vh, h, target = v)
        v.add_col_vec(w_v)
        v.apply_sigmoid()

        cm.dot(w_vh.T, v, target = h)
        h.add_col_vec(w_h)
        h.apply_sigmoid()

        wu_vh.subtract_dot(v, h.T)
        wu_v.add_sums(v, axis = 1, mult = -1.)
        wu_h.add_sums(h, axis = 1, mult = -1.)

        # update weights
        w_vh.add_mult(wu_vh, epsilon/batch_size)
        w_v.add_mult(wu_v, epsilon/batch_size)
        w_h.add_mult(wu_h, epsilon/batch_size)

        # calculate reconstruction error
        v.subtract(v_true)
        err.append(v.euclid_norm()**2/(num_vis*batch_size))

    print("Mean squared error: %f" % np.mean(err))
    print("Time: %f" % (time.time() - start_time))

w_vh.copy_to_host()
util.save('weights.dat', 'w_vh', {'w_vh': w_vh.numpy_array})

cm.cublas_shutdown()
