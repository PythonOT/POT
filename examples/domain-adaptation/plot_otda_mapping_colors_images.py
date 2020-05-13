# -*- coding: utf-8 -*-
"""
=====================================================
OT for image color adaptation with mapping estimation
=====================================================

OT for domain adaptation with image color adaptation [6] with mapping
estimation [8].

[6] Ferradans, S., Papadakis, N., Peyre, G., & Aujol, J. F. (2014). Regularized
discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

[8] M. Perrot, N. Courty, R. Flamary, A. Habrard, "Mapping estimation for
discrete optimal transport", Neural Information Processing Systems (NIPS), 2016.

"""

# Authors: Remi Flamary <remi.flamary@unice.fr>
#          Stanislas Chambon <stan.chambon@gmail.com>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pylab as pl
import ot

r = np.random.RandomState(42)


def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


##############################################################################
# Generate data
# -------------

# Loading images
I1 = pl.imread('../../data/ocean_day.jpg').astype(np.float64) / 256
I2 = pl.imread('../../data/ocean_sunset.jpg').astype(np.float64) / 256


X1 = im2mat(I1)
X2 = im2mat(I2)

# training samples
nb = 1000
idx1 = r.randint(X1.shape[0], size=(nb,))
idx2 = r.randint(X2.shape[0], size=(nb,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]


##############################################################################
# Domain adaptation for pixel distribution transfer
# -------------------------------------------------

# EMDTransport
ot_emd = ot.da.EMDTransport()
ot_emd.fit(Xs=Xs, Xt=Xt)
transp_Xs_emd = ot_emd.transform(Xs=X1)
Image_emd = minmax(mat2im(transp_Xs_emd, I1.shape))

# SinkhornTransport
ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
Image_sinkhorn = minmax(mat2im(transp_Xs_sinkhorn, I1.shape))

ot_mapping_linear = ot.da.MappingTransport(
    mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
ot_mapping_linear.fit(Xs=Xs, Xt=Xt)

X1tl = ot_mapping_linear.transform(Xs=X1)
Image_mapping_linear = minmax(mat2im(X1tl, I1.shape))

ot_mapping_gaussian = ot.da.MappingTransport(
    mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10, verbose=True)
ot_mapping_gaussian.fit(Xs=Xs, Xt=Xt)

X1tn = ot_mapping_gaussian.transform(Xs=X1)  # use the estimated mapping
Image_mapping_gaussian = minmax(mat2im(X1tn, I1.shape))


##############################################################################
# Plot original images
# --------------------

pl.figure(1, figsize=(6.4, 3))
pl.subplot(1, 2, 1)
pl.imshow(I1)
pl.axis('off')
pl.title('Image 1')

pl.subplot(1, 2, 2)
pl.imshow(I2)
pl.axis('off')
pl.title('Image 2')
pl.tight_layout()


##############################################################################
# Plot pixel values distribution
# ------------------------------

pl.figure(2, figsize=(6.4, 5))

pl.subplot(1, 2, 1)
pl.scatter(Xs[:, 0], Xs[:, 2], c=Xs)
pl.axis([0, 1, 0, 1])
pl.xlabel('Red')
pl.ylabel('Blue')
pl.title('Image 1')

pl.subplot(1, 2, 2)
pl.scatter(Xt[:, 0], Xt[:, 2], c=Xt)
pl.axis([0, 1, 0, 1])
pl.xlabel('Red')
pl.ylabel('Blue')
pl.title('Image 2')
pl.tight_layout()


##############################################################################
# Plot transformed images
# -----------------------

pl.figure(2, figsize=(10, 5))

pl.subplot(2, 3, 1)
pl.imshow(I1)
pl.axis('off')
pl.title('Im. 1')

pl.subplot(2, 3, 4)
pl.imshow(I2)
pl.axis('off')
pl.title('Im. 2')

pl.subplot(2, 3, 2)
pl.imshow(Image_emd)
pl.axis('off')
pl.title('EmdTransport')

pl.subplot(2, 3, 5)
pl.imshow(Image_sinkhorn)
pl.axis('off')
pl.title('SinkhornTransport')

pl.subplot(2, 3, 3)
pl.imshow(Image_mapping_linear)
pl.axis('off')
pl.title('MappingTransport (linear)')

pl.subplot(2, 3, 6)
pl.imshow(Image_mapping_gaussian)
pl.axis('off')
pl.title('MappingTransport (gaussian)')
pl.tight_layout()

pl.show()
