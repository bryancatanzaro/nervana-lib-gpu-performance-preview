# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
import numpy as np
import pycuda.driver as drv
from flexpt_array import Flexpt
from pycuda.autoinit import context
import struct

fp = Flexpt(kernel_set="fgemm_float32_wide64", calc_partials=False)

op     = "nt"  # n == not transpose, t == transpose
m      = 4096 
n      = 4096
k      = 4096
repeat = 50
iwlA   = 15
iwlB   = 15

if op == "nt":
    dim1 = (k,m)
    dim2 = (k,n)
elif op == "nn":
    dim1 = (m,k)
    dim2 = (k,n)
elif op == "tn":
    dim1 = (m,k)
    dim2 = (n,k)

A1 = np.random.randint(0x0, 0x7fff, size=dim1).astype(np.int64)
B1 = np.random.randint(0x0, 0x7fff, size=dim2).astype(np.int64)

A2 = fp.array(A1.astype(np.int16), iwlA)
B2 = fp.array(B1.astype(np.int16), iwlB)

# pick a reasonable output integer word length
iwlC = ((struct.unpack('I',struct.pack('f',float(0x7fff * 0x7fff * k / 2)))[0] & 0x7f800000) >> 23)-126

C2 = fp.empty((m,n), iwlC)

start = drv.Event()
end   = drv.Event()

start.record()

for r in range(repeat):
    if   op == 'nt':
        fp.dot(A2.T, B2, C2)
    elif op == 'nn':
        fp.dot(A2, B2, C2)
    elif op == 'tn':
        fp.dot(A2, B2.T, C2)

end.record()
end.synchronize()

msecs = end.time_since(start) / repeat
gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
print 'GFLOPS: ', gflops
