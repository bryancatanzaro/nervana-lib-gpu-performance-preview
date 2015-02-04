# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
A simple demo script for convolution kernels fprop, bprop, update.

We use cuDNN paper[*] notation for the various dimensions:
  N - number of images in mini-batch
  C - number of input feature maps
  K - number of output feature maps
  (D, H, W) - depth, height and width of input image
  (T, R, S) - depth, height and width of filter kernels
  padding_{x,y,z} - zero padding
  strides_{x,y,z} - filter striding
  upscale_{x,y,z} - upscaling

[*] Chetlur et al. 'cuDNN: Efficient primitives for deep learning.' arXiv:1410.0759
"""
import numpy as np
import struct
import pycuda.driver as drv
from flexpt_array import Flexpt
import pycuda.autoinit 

# select kernel set (just one in this release, more later)
fp = Flexpt(kernel_set="fgemm_float32_wide64", bench=True)

# set dims for layer 5 of Alexnet
N,C,K = (128,192,384)
D,H,W = (1,13,13)
T,R,S = (1,3,3)

# set padding, stride and upscale
padding_z, padding_y, padding_x = (0,0,0)
strides_z, strides_y, strides_x = (1,1,1)
upscale_z, upscale_y, upscale_x = (1,1,1)

# set input integer word length
iwl = 15

# input dimensions
dimI = (C,D,H,W,N)
dimF = (C,T,R,S,K)

# output dimensions (taking into account for scale, pad, stride)
M = (D*upscale_z - T + 1 + 2*padding_z) // strides_z
P = (H*upscale_y - R + 1 + 2*padding_y) // strides_y
Q = (W*upscale_x - S + 1 + 2*padding_x) // strides_x

padding = (padding_z, padding_y, padding_x)
strides = (strides_z, strides_y, strides_x)
upscale = (upscale_z, upscale_y, upscale_x)

dimO = (K,M,P,Q,N)

# create random input into kernels and allocate output and bit widths
# NOTE: higher entropy will force chip to lower clocks.
I = np.random.randint(0x0, 0x7fff, size=dimI).astype(np.int64)
F = np.random.randint(0x0, 0x7fff, size=dimF).astype(np.int64)
E = np.random.randint(0x0, 0x7fff, size=dimO).astype(np.int64)

# copy to device
devI = fp.array(I, iwl)
devF = fp.array(F, iwl)
devE = fp.array(E, iwl)

# set output bit widths at approximately mean scaling
def scale(n,q):
    return ((struct.unpack('I',struct.pack('f',float(0x7fff**2 * n) / q )))[0] >> 23)-126

iwlO = scale(C*T*R*S, 2)
iwlB = scale(K*T*R*S, 4)
iwlU = scale(N*M*P*Q, 4)

# allocate output 
devO = fp.empty(dimO, iwlO)
devB = fp.zeros(dimI, iwlB)
devU = fp.zeros(dimF, iwlU)
args = dict(padding=padding, strides=strides, upscale=upscale, repeat=100)

# perform convolutions
print 'Warming up'
fp.fprop_conv(devI, devF, devO, strides=strides, upscale=upscale, repeat=10)  # spin up clock

print 'Starting'
fp.fprop_conv(devI, devF, devO, **args)
fp.bprop_conv(devF, devE, devB, **args)
fp.update_conv(devI, devE, devU, **args)
print 'Done'


