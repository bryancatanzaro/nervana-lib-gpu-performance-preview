# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
A benchmark script for Soumith and his page:

https://github.com/soumith/convnet-benchmarks
"""
import numpy as np
import struct
import pycuda.driver as drv
from flexpt_array import Flexpt
import pycuda.autoinit 

fp = Flexpt(kernel_set="fgemm_float32_wide64", bench=True)

def scale(n,q):
    return ((struct.unpack('I',struct.pack('f',float(0x7fff**2 * n) / q )))[0] >> 23)-126

def go(N,C,K,D,H,W,T,R,S):
    # input dimensions
    dimI = (C,D,H,W,N)
    dimF = (C,T,R,S,K)

    # set padding, stride and upscale
    padding_z, padding_y, padding_x = (0,0,0)
    strides_z, strides_y, strides_x = (1,1,1)
    upscale_z, upscale_y, upscale_x = (1,1,1)

    # output dimensions (taking into account for scale, pad, stride)
    M = (D*upscale_z - T + 1 + 2*padding_z) // strides_z
    P = (H*upscale_y - R + 1 + 2*padding_y) // strides_y
    Q = (W*upscale_x - S + 1 + 2*padding_x) // strides_x

    padding = (padding_z, padding_y, padding_x)
    strides = (strides_z, strides_y, strides_x)
    upscale = (upscale_z, upscale_y, upscale_x)

    dimO = (K,M,P,Q,N)

    # create random input into kernels and allocate output and bit widths
    # NOTE: higher entropy will force chip to lower clocks and lower performance.
    I = np.random.randint(0x0, 0x7fff, size=dimI).astype(np.int64)
    F = np.random.randint(0x0, 0x7fff, size=dimF).astype(np.int64)
    E = np.random.randint(0x0, 0x7fff, size=dimO).astype(np.int64)

    # input integer word length
    iwl = 15
    
    # copy to device
    devI = fp.array(I, iwl)
    devF = fp.array(F, iwl)
    devE = fp.array(E, iwl)

    iwlO = scale(C*T*R*S, 2)
    iwlB = scale(K*T*R*S, 4)
    iwlU = scale(N*M*P*Q, 4)

    # allocate output 
    devO = fp.empty(dimO, iwlO)
    devB = fp.zeros(dimI, iwlB)
    devU = fp.zeros(dimF, iwlU)
    args = dict(padding=padding, strides=strides, upscale=upscale, repeat=1)
    
    # perform convolutions and get timings
    f, _ = fp.fprop_conv(devI, devF, devO, **args)
    b, _ = fp.bprop_conv(devF, devE, devB, **args)
    u, _ = fp.update_conv(devI, devE, devU, **args)

    return f, b, u

def winners():
    """Benchmarks for popular imagenet models"""
    # [TODO] Need to look up dims for models on Soumith's site
    #   ignore pooling and softmax layers
    pass
    

def layerwise(plot=False):
    r = []
    
    # layer 1
    N,C,K = (128,3,96)
    D,H,W = (1,128,128)
    T,R,S = (1,11,11)

    r.append(go(N,C,K,D,H,W,T,R,S))

    # layer 2
    N,C,K = (128,64,128)
    D,H,W = (1,64,64)
    T,R,S = (1,9,9)

    r.append(go(N,C,K,D,H,W,T,R,S))

    # layer 3
    N,C,K = (128,128,128)
    D,H,W = (1,32,32)
    T,R,S = (1,9,9)

    r.append(go(N,C,K,D,H,W,T,R,S))

    # layer 4
    N,C,K = (128,128,128)
    D,H,W = (1,16,16)
    T,R,S = (1,7,7)

    r.append(go(N,C,K,D,H,W,T,R,S))

    # layer 5
    N,C,K = (128,384,384)
    D,H,W = (1,13,13)
    T,R,S = (1,3,3)

    r.append(go(N,C,K,D,H,W,T,R,S))

    r = np.array(r)
    f = r[:,0]
    b = r[:,1]
    u = r[:,2]

    # cherry-picked benchmark numbers (from Soumith's page)
    cf = np.array([63., 72., 30., 9., 17.])
    cb = np.array([86., 230., 82., 8., 16.])
    cu = np.array([199., 107., 36., 9., 21.])

    # cudnn r2 numbers (from Soumith's page)
    c2f = np.array([90., 218, 79., 9., 17.])
    c2b = np.array([91., 344., 130., 15., 20.])
    c2u = np.array([98., 262., 100, 12, 27.])
    
    np.set_printoptions(precision=0)
    print 'Our numbers (Layers 1-5 and total in msecs, lower is better):'
    print 
    print '  forward:    %s  total=%d' % (f, f.sum())
    print '  backward:   %s  total=%d' % (b+u, (b+u).sum())
    print '  gradInput:  %s  total=%d' % (b, b.sum())
    print '  gradWeight: %s  total=%d' % (u, u.sum())

    print
    print 'For comparison, cuDNN R2 reference numbers, from'
    print '  https://github.com/soumith/convnet-benchmarks (accessed 2/2/15):'
    print
    print '  forward:    %s  total=%d' % (c2f, c2f.sum())
    print '  backward:   %s  total=%d' % (c2b+c2u, (c2b+c2u).sum())
    print '  gradInput:  %s  total=%d' % (c2b, c2b.sum())
    print '  gradWeight: %s  total=%d' % (c2u, c2u.sum())

if __name__ == '__main__':
    layerwise()

"""
Sample run:

  forward: [  41.  112.   42.    5.    9.]  total=208
  backward: [ 139.  240.   87.    9.   18.]  total=493
  gradInput: [  81.  124.   46.    5.    9.]  total=264
  gradWeight: [  58.  116.   41.    4.    9.]  total=228
"""
