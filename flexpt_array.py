# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
An array class wrapper for GEMM and convolution kernels.

Much of the array class is based on pycuda's GPUArray [*].

* Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul
  Ivanov, Ahmed Fasih, PyCUDA and PyOpenCL: A scripting-based approach
  to GPU run-time code generation, Parallel Computing, Volume 38,
  Issue 3, March 2012, Pages 157-174.
"""

import numpy as np
import struct
import re
from operator import mul
import pycuda.driver as drv
from pytools import memoize, memoize_method
from pycuda.compyte.array import (
        as_strided as _as_strided,
        f_contiguous_strides as _f_contiguous_strides,
        c_contiguous_strides as _c_contiguous_strides,
        ArrayFlags as _ArrayFlags)
import flexpt_ew

class FlexptArray(object):

    def __init__(self, backend, shape, iwl, allocator=drv.mem_alloc,
            base=None, gpudata=None, strides=None, is_trans=False, order="C"):
        
        dtype = np.dtype(np.int16)

        try:
            size = 1
            for dim in shape:
                size *= dim
        except TypeError:
            assert isinstance(shape, (int, long, np.integer))
            size  = shape
            shape = (shape,)

        if isinstance(size, np.integer):
            size = np.asscalar(size)

        if strides is None:
            if order == "F":
                strides = _f_contiguous_strides(dtype.itemsize, shape)
            elif order == "C":
                strides = _c_contiguous_strides(dtype.itemsize, shape)
            else:
                raise ValueError("invalid order: %s" % order)
        else:
            strides = tuple(strides)

        self.backend   = backend
        self.base      = base
        self.shape     = shape
        self.iwl       = iwl
        self.strides   = strides
        self.size      = size
        self.dtype     = dtype
        self.nbytes    = dtype.itemsize * size
        self.allocator = allocator
        self.is_trans  = is_trans

        if gpudata is None:
            if size:
                self.gpudata = allocator(self.nbytes)
            else:
                self.gpudata = None

            assert base is None
        else:
            self.gpudata = gpudata

    def __str__(self):
        return "FlexptArray shape:%s strides:%s is_trans:%s" % (self.shape, self.strides, self.is_trans)

    def __repr__(self):
        return "FlexptArray"

    def __int__(self):
        return int(selt.gpudata)

    def __len__(self):
        """Return the size of the leading dimension of self."""
        if len(self.shape):
            return self.shape[0]
        else:
            return 0

    @property
    @memoize_method
    def flags(self):
        return _ArrayFlags(self)
  
    def set(self, ary):
        assert ary.dtype   == np.int16
        assert ary.size    == self.size
        assert ary.strides == self.strides
        assert self.flags.forc

        if self.size:
            drv.memcpy_htod(self.gpudata, ary)

        return self

    def get(self, ary=None, astype=None, pagelocked=False):
        if ary is None:
            if pagelocked:
                ary = drv.pagelocked_empty(self.shape, self.dtype)
            else:
                ary = np.empty(self.shape, self.dtype)

            ary = _as_strided(ary, strides=self.strides)
        else:
            assert ary.size == self.size
            assert ary.dtype == self.dtype
            assert ary.flags.forc

        assert self.flags.forc, "Array in get() must be contiguous"

        if self.size:
            drv.memcpy_dtoh(ary, self.gpudata)

        if astype is not None:
            ary = ary.astype(astype) * 2 ** (self.iwl - 15)

        return ary

    def __getitem__(self, index):
        """
        return a sliced view of an array
        """
        if not isinstance(index, tuple):
            index = (index,)

        new_shape = []
        new_offset = 0
        new_strides = []

        seen_ellipsis = False

        index_axis = 0
        array_axis = 0
        while index_axis < len(index):
            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            if isinstance(index_entry, slice):
                start, stop, idx_stride = index_entry.indices(
                        self.shape[array_axis])

                array_stride = self.strides[array_axis]

                new_shape.append((stop-start)//idx_stride)
                new_strides.append(idx_stride*array_stride)
                new_offset += array_stride*start

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, np.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError(
                            "subindex in axis %d out of range" % index_axis)

                new_offset += self.strides[array_axis]*index_entry

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError(
                            "more than one ellipsis not allowed in index")
                seen_ellipsis = True

            else:
                raise IndexError("invalid subindex in axis %d" % index_axis)

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return self.__class__(
                backend   = self.backend,
                shape     = tuple(new_shape),
                iwl       = self.iwl,
                allocator = self.allocator,
                base      = self,
                gpudata   = int(self.gpudata)+new_offset,
                strides   = tuple(new_strides))

    def _assign(self, value):
        
        if isinstance(value, (int, float)):

            # if we have a c or f contiguous array, then use the speedy driver kernel
            if self.flags.forc and float(value) >= 0:
                drv.memset_d16(self.gpudata, Flexpt.flex_from_native(value,self.iwl), self.size)
            # otherwise use our copy kerel
            else:
                OpTreeNode.build("copy", value, None, out=self)

        elif isinstance(value, FlexptArray):
            if self.flags.forc and value.flags.forc and self.iwl == value.iwl:
                drv.memcpy_dtod(self.gpudata, value.gpudata, self.nbytes)
            else:
                OpTreeNode.build("copy", value, None, out=self)
        
        elif isinstance(value, OpTreeNode):
            value.execute(out=self)

        else:
            raise TypeError("Invalid type for assignment: %s" % type(value))

        return self

    def __setitem__(self, index, value):

        self.__getitem__(index)._assign(value)

    def fill(self, value):

        return self._assign(value)

    def copy(self, a):

        return self._assign(a)

    def reshape(self, *shape):
        
        if isinstance(shape[0], tuple) or isinstance(shape[0], list):
            shape = tuple(shape[0])

        if shape == self.shape:
            return self

        size = reduce(lambda x, y: x * y, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        if not self.flags.forc:
            raise TypeError("reshaping of non-contigous arrays is not yet supported")

        return self.__class__(
                backend   = self.backend,
                shape     = shape,
                iwl       = self.iwl,
                allocator = self.allocator,
                base      = self,
                gpudata   = self.gpudata,
                strides   = _c_contiguous_strides(self.dtype.itemsize, shape))

    @property
    def T(self):
        """
        return a transposed view
        """
        return self.__class__(
                backend   = self.backend,
                shape     = self.shape[::-1],
                iwl       = self.iwl,
                allocator = self.allocator,
                base      = self,
                gpudata   = self.gpudata,
                strides   = self.strides[::-1],
                is_trans  = not self.is_trans)

    def __add__      (self, other): return OpTreeNode.build("add", self, other)
    def __sub__      (self, other): return OpTreeNode.build("sub", self, other)
    def __mul__      (self, other): return OpTreeNode.build("mul", self, other)
    def __div__      (self, other): return OpTreeNode.build("div", self, other)
    def __truediv__  (self, other): return OpTreeNode.build("div", self, other)
    def __pow__      (self, other): return OpTreeNode.build("pow", self, other)
    def __radd__     (self, other): return OpTreeNode.build("add", other, self)
    def __rsub__     (self, other): return OpTreeNode.build("sub", other, self)
    def __rmul__     (self, other): return OpTreeNode.build("mul", other, self)
    def __rdiv__     (self, other): return OpTreeNode.build("div", other, self)
    def __rtruediv__ (self, other): return OpTreeNode.build("div", other, self)
    def __rpow__     (self, other): return OpTreeNode.build("pow", other, self)
    def __eq__       (self, other): return OpTreeNode.build("eq",  self, other)
    def __ne__       (self, other): return OpTreeNode.build("ne",  self, other)
    def __lt__       (self, other): return OpTreeNode.build("lt",  self, other)
    def __le__       (self, other): return OpTreeNode.build("le",  self, other)
    def __gt__       (self, other): return OpTreeNode.build("gt",  self, other)
    def __ge__       (self, other): return OpTreeNode.build("ge",  self, other)
    def __abs__      (self):        return OpTreeNode.build("abs", self,  None)
    def __neg__      (self):        return OpTreeNode.build("neg", self,  None)

    def __iadd__     (self, other): return OpTreeNode.build("add", self, other, out=self)
    def __isub__     (self, other): return OpTreeNode.build("sub", self, other, out=self)
    def __imul__     (self, other): return OpTreeNode.build("mul", self, other, out=self)
    def __idiv__     (self, other): return OpTreeNode.build("div", self, other, out=self)
    def __itruediv__ (self, other): return OpTreeNode.build("div", self, other, out=self)
    def __ipow__     (self, other): return OpTreeNode.build("pow", self, other, out=self)



class Flexpt(object):

    def __init__(self, kernel_set="fgemm_int64_wide32", locks=1024, calc_partials=True, bench=False):

        m = re.search( r'wide(\d+)', kernel_set)
        if m:
            self.width  = int(m.group(1))
        else:
            raise ValueError("Invalid kernel_set")
        
        self.locks   = locks
        self.module  = drv.module_from_file("kernels/" + kernel_set + ".cubin")
        self.mode    = 0 if calc_partials else 4
        self.fgemm   = dict()
        for op in ("nt", "nn", "tn"):
            mod = self.module.get_function(kernel_set + "_" + op)
            mod.prepare("PPPIIIIIIHH")
            self.fgemm[op] = mod

        fprop_conv = self.module.get_function("fprop_conv_float32_K64N64T64")
        fprop_conv.prepare("PPPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        self.fgemm["fprop_conv"] = fprop_conv

        bprop_conv = self.module.get_function("bprop_conv_float32_CRST64N64T64")
        bprop_conv.prepare("PPPPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        self.fgemm["bprop_conv"] = bprop_conv

        udpate_conv = self.module.get_function("update_conv_float32_CRST64K64T64")
        udpate_conv.prepare("PPPPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        self.fgemm["update_conv"] = udpate_conv

        self.gpulock = drv.mem_alloc(locks*4)
        drv.memset_d32(self.gpulock, 0, locks)

        self.bench = bench
        if bench:
            self.start = drv.Event()
            self.end   = drv.Event()

    def empty(self, shape, iwl=None, allocator=drv.mem_alloc, order="C"):
        """
        allocate the space for a FlexptArray
        """

        return FlexptArray(self, shape, iwl, allocator=allocator, order=order)

    def array(self, ary, iwl=None, allocator=drv.mem_alloc):
        """
        converts a numpy array to a FlexptArray
        """

        if iwl is None:
            ary, iwl = self.flex_from_native(ary)

        elif ary.dtype != np.int16:
            ary = self.flex_from_native(ary, iwl)

        return FlexptArray(self, ary.shape, iwl, allocator=allocator, strides=ary.strides).set(ary)

    def zeros(self, shape, iwl, allocator=drv.mem_alloc, order="C"):
        """
        Returns an array of the given shape and dtype filled with 0's.
        """

        result = FlexptArray(self, shape, iwl, allocator, order=order)

        drv.memset_d16(result.gpudata, 0, result.size)

        return result

    def ones(self, shape, iwl, allocator=drv.mem_alloc, order="C"):
        """
        Returns an array of the given shape and dtype filled with 0's.
        """

        result = FlexptArray(self, shape, iwl, allocator, order=order)

        drv.memset_d16(result.gpudata, self.flex_from_native(1,iwl), result.size)

        return result

    def empty_like(self, other_ary):
        """
        Returns an array with the same params as another
        """
        return FlexptArray(self, other_ary.shape, other_ary.iwl, other_ary.allocator)

    def dot(self, A, B, out, iwl=None):

        # one dimention must be contiguous
        assert min(A.strides) == 2
        assert min(B.strides) == 2
        assert min(out.strides) == 2

        opA = 't' if A.is_trans else 'n'  # A.strides[0] < A.strides[1]
        opB = 't' if B.is_trans else 'n'  # B.strides[0] < B.strides[1]
        op  = opB + opA

        # TODO: use nn and swapp A and B if we need tt op (C is transposed though, I think)
        assert op != "tt"

        m = A.shape[0]
        n = B.shape[1]
        k = A.shape[1]

        assert m == out.shape[0]
        assert n == out.shape[1]
        assert k == B.shape[0]

        lda = max(A.strides) // 2
        ldb = max(B.strides) // 2
        ldc = max(out.strides) // 2

        gridX = n // self.width + (n % self.width != 0)
        gridY = m // self.width + (m % self.width != 0)

        if iwl is not None: 
            out.iwl = iwl

        scale = 15 + out.iwl - (A.iwl + B.iwl)

        if self.bench:
            self.start.record()

        self.fgemm[op].prepared_call(
            (gridX,gridY,1), (64,1,1),
            B.gpudata, A.gpudata, out.gpudata,
            ldb, lda, ldc,
            n, m, k, 
            scale, self.mode)

        if self.bench:
            self.end.record()
            self.end.synchronize()
            msecs = self.end.time_since(self.start)
            gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
            print "%s (%5d, %5d, %5d): %9.6f msecs %6.1f gflops" % (op,m,n,k,msecs,gflops)
            return (msecs, gflops)

        return out

    def update_conv(self, I, E, F, padding=(0,0,0), strides=(1,1,1), upscale=(1,1,1), iwl=None, repeat=1):

        # I dims: C,D,H,W,N
        # F dims: C,T,R,S,K
        # E dims: K,M,P,Q,N
        assert len(I.shape) == len(F.shape) == len(E.shape), "I, F, and E must be of same dimensions"

        dims = len(I.shape) - 2
        assert 1 <= dims <= 3, "The image data must be between 1 and 3 dimensions" 

        N = I.shape[-1]
        K = F.shape[-1]
        assert N % 8 == 0, "N dim must be multiple of 8"
        assert K % 8 == 0, "K dim must be multiple of 8"
        assert N == E.shape[-1], "I and E minibatch sizes do not match"

        C = I.shape[0]
        assert C == F.shape[0], "Channel counts of I and F do not match"
        assert K == E.shape[0], "Filter banks and output channels do not match"

        if iwl is not None: 
            F.iwl = iwl
        scale = 15 + F.iwl - (I.iwl + E.iwl)

        DHW, TRS, MPQ, pad, std, scl = ([],[],[],[],[],[])

        # initialize any unspecified leading dimensions
        for i in range(3 - dims):
            for lst in (DHW, TRS, MPQ, std, scl):
                lst.append(1)
            pad.append(0)

        # fill in the rest
        DHW.extend(I.shape[1:dims+1])
        TRS.extend(F.shape[1:dims+1])
        MPQ.extend(E.shape[1:dims+1])
        pad.extend(padding[-dims:])
        std.extend(strides[-dims:])
        scl.extend(upscale[-dims:])

        for i in range(len(DHW)):
            if __debug__:
                dim = (DHW[i]*scl[i] - TRS[i] + 1 + 2*pad[i]) // std[i]
            assert MPQ[i] == dim, "Output dim=%d value=%d does not match calculated value=%d" % (i, MPQ[i], dim)

            scl[i] = magic32(MPQ[i]+TRS[i]-pad[i]-2, scl[i])

        W    = DHW[2]
        WN   = W*N
        HWN  = DHW[1]*WN
        DHWN = DHW[0]*HWN
        S    = TRS[2]
        RS   = TRS[1]*S
        RST  = TRS[0]*RS
        CRST = C*RST
        Q    = MPQ[2]
        P    = MPQ[1]
        PM   = MPQ[0]*P
        QPM  = Q*PM

        magic_RST = magic32(CRST+8, RST)
        magic_RS  = magic32(RST+32, RS )
        magic_S   = magic32(RS+32,  S  )
        magic_P   = magic32(PM,     P  )
        
        gridI = CRST // 64 + (CRST % 64 != 0)
        gridE = K // 64 + (K % 64 != 0)

        kernel_args = flatten([
            F.gpudata, I.gpudata, E.gpudata, self.gpulock, scale, 
            N, K, DHW, WN, HWN, DHWN,
            C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S, 
            pad, std, scl,
            Q, P, Q*N, P*Q*N, QPM*N, magic_P])

        if self.bench:
            self.start.record()

        for r in range(repeat):
            self.fgemm["update_conv"].prepared_call(
                (gridI, gridE, PM), (64,1,1), *kernel_args)

        if self.bench:
            self.end.record()
            self.end.synchronize()
            msecs = self.end.time_since(self.start) / repeat
            gflops = (QPM * K * N * CRST * 2.0) / (msecs * 1000000.0)
            print "update_conv NCK: (%d,%d,%d) DHW:%s TRS:%s MPQ:%s: %9.6f msecs %6.1f gflops" % \
                (N, C, K, DHW, TRS, MPQ, msecs, gflops)
            return (msecs, gflops)


    def fprop_conv(self, I, F, O, padding=(0,0,0), strides=(1,1,1), upscale=(1,1,1), iwl=None, repeat=1):

        # I dims: C,D,H,W,N
        # F dims: C,T,R,S,K
        # O dims: K,M,P,Q,N
        assert len(I.shape) == len(F.shape) == len(O.shape), "I, F, and O must be of same dimensions"

        dims = len(I.shape) - 2
        assert 1 <= dims <= 3, "The image data must be between 1 and 3 dimensions" 

        N = I.shape[-1]
        K = F.shape[-1]
        assert N % 8 == 0, "N dim must be multiple of 8"
        assert K % 8 == 0, "K dim must be multiple of 8"
        assert N == O.shape[-1], "I and O minibatch sizes do not match"

        C = I.shape[0]
        assert C == F.shape[0], "Channel counts of I and F do not match"
        assert K == O.shape[0], "Filter banks and output channels do not match"

        if iwl is not None: 
            O.iwl = iwl
        scale = 15 + O.iwl - (I.iwl + F.iwl)

        DHW, TRS, MPQ, pad, std, scl = ([],[],[],[],[],[])

        # initialize any unspecified leading dimensions
        for i in range(3 - dims):
            for lst in (DHW, TRS, MPQ, std, scl):
                lst.append(1)
            pad.append(0)

        # fill in the rest
        DHW.extend(I.shape[1:dims+1])
        TRS.extend(F.shape[1:dims+1])
        MPQ.extend(O.shape[1:dims+1])
        pad.extend(padding[-dims:])
        std.extend(strides[-dims:])
        scl.extend(upscale[-dims:])

        for i in range(len(DHW)):
            if __debug__:
                dim = (DHW[i]*scl[i] - TRS[i] + 1 + 2*pad[i]) // std[i]
            assert MPQ[i] == dim, "Output dim=%d value=%d does not match calculated value=%d" % (i, MPQ[i], dim)

            scl[i] = magic32(MPQ[i]+TRS[i]-pad[i]-2, scl[i])

        W    = DHW[2]
        WN   = W*N
        HWN  = DHW[1]*WN
        DHWN = DHW[0]*HWN
        S    = TRS[2]
        RS   = TRS[1]*S
        RST  = TRS[0]*RS
        CRST = C*RST
        Q    = MPQ[2]
        PQ   = MPQ[1]*Q
        PQM  = MPQ[0]*PQ

        magic_RST = magic32(CRST+8, RST)
        magic_RS  = magic32(RST+32, RS)
        magic_S   = magic32(RS+32,  S)
        magic_PQ  = magic32(PQM, PQ)
        magic_Q   = magic32(PQ,  Q)
        
        gridI = N // 64 + (N % 64 != 0)
        gridF = K // 64 + (K % 64 != 0)

        share = (RST // 32 + (RST % 32 != 0)) * 32 * 4

        kernel_args = flatten([
            O.gpudata, I.gpudata, F.gpudata, scale, 
            N, K, DHW, WN, HWN, DHWN,
            C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S, 
            pad, std, scl,
            Q, PQ, Q*N, PQ*N, PQM*N, magic_PQ, magic_Q])

        if self.bench:
            self.start.record()

        for r in range(repeat):
            self.fgemm["fprop_conv"].prepared_call(
                (gridF, gridI, PQM), (64,1,1), *kernel_args, shared_size=share)

        if self.bench:
            self.end.record()
            self.end.synchronize()
            msecs = self.end.time_since(self.start) / repeat
            gflops = (PQM * K * N * CRST * 2.0) / (msecs * 1000000.0)
            print "fprop_conv  NCK: (%d,%d,%d) DHW:%s TRS:%s MPQ:%s: %9.6f msecs %6.1f gflops" % \
                (N, C, K, DHW, TRS, MPQ, msecs, gflops)
            return (msecs, gflops)

    def bprop_conv(self, F, E, I, padding=(0,0,0), strides=(1,1,1), upscale=(1,1,1), iwl=None, repeat=1):

        # I dims: C,D,H,W,N
        # F dims: C,T,R,S,K
        # O dims: K,M,P,Q,N
        assert len(I.shape) == len(F.shape) == len(E.shape), "I, F, and E must be of same dimensions"

        dims = len(I.shape) - 2
        assert 1 <= dims <= 3, "The image data must be between 1 and 3 dimensions" 

        N = I.shape[-1]
        K = F.shape[-1]
        assert N % 8 == 0, "N dim must be multiple of 8"
        assert K % 8 == 0, "K dim must be multiple of 8"
        assert N == E.shape[-1], "I and E minibatch sizes do not match"

        C = I.shape[0]
        assert C == F.shape[0], "Channel counts of I and F do not match"
        assert K == E.shape[0], "Filter banks and output channels do not match"

        if iwl is not None: 
            I.iwl = iwl
        scale = 15 + I.iwl - (F.iwl + E.iwl)

        DHW, TRS, MPQ, pad, std, scl = ([],[],[],[],[],[])

        # initialize any unspecified leading dimensions
        for i in range(3 - dims):
            for lst in (DHW, TRS, MPQ, std, scl):
                lst.append(1)
            pad.append(0)

        # fill in the rest
        DHW.extend(I.shape[1:dims+1])
        TRS.extend(F.shape[1:dims+1])
        MPQ.extend(E.shape[1:dims+1])
        pad.extend(padding[-dims:])
        std.extend(strides[-dims:])
        scl.extend(upscale[-dims:])

        for i in range(len(DHW)):
            if __debug__:
                dim = (DHW[i]*scl[i] - TRS[i] + 1 + 2*pad[i]) // std[i]
            assert MPQ[i] == dim, "Output dim=%d value=%d does not match calculated value=%d" % (i, MPQ[i], dim)

            scl[i] = magic32(MPQ[i]+TRS[i]-pad[i]-2, scl[i])

        W    = DHW[2]
        WN   = W*N
        HWN  = DHW[1]*WN
        DHWN = DHW[0]*HWN
        S    = TRS[2]
        RS   = TRS[1]*S
        RST  = TRS[0]*RS
        CRST = C*RST
        Q    = MPQ[2]
        PQ   = MPQ[1]*Q
        PQM  = MPQ[0]*PQ

        magic_RST = magic32(CRST+8, RST)
        magic_RS  = magic32(RST+32, RS)
        magic_S   = magic32(RS+32,  S)
        magic_PQ  = magic32(PQM, PQ)
        magic_Q   = magic32(PQ,  Q)

        lockC = len(bin(128 // RST))-2
        
        gridF = CRST // 64 + (CRST % 64 != 0)
        gridE = N    // 64 + (N    % 64 != 0)

        share = (RST // 32 + (RST % 32 != 0)) * 32 * 4

        kernel_args = flatten([
            I.gpudata, F.gpudata, E.gpudata, self.gpulock, scale, 
            N, K, DHW, WN, HWN, DHWN,
            C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S, 
            pad, std, scl,
            Q, PQ, Q*N, PQ*N, PQM*N, magic_PQ, magic_Q, PQM*N*8*2, magic_RST[1]+lockC])

        if self.bench:
            self.start.record()

        for r in range(repeat):
            self.fgemm["bprop_conv"].prepared_call(
                (gridF, gridE, PQM), (64,1,1), *kernel_args, shared_size=share)

        if self.bench:
            self.end.record()
            self.end.synchronize()
            msecs = self.end.time_since(self.start) / repeat
            gflops = (PQM * K * N * CRST * 2.0) / (msecs * 1000000.0)
            print "bprop_conv  NCK: (%d,%d,%d) DHW:%s TRS:%s MPQ:%s: %9.6f msecs %6.1f gflops" % \
                (N, C, K, DHW, TRS, MPQ, msecs, gflops)
            return (msecs, gflops)

    def clip(self, a, a_min, a_max, out=None):
        
        return OpTreeNode.build("min", OpTreeNode.build("max", a, a_min), a_max, out=out)

    def maximum(self, a, b, out=None): return OpTreeNode.build("max", a, b, out=out)
    def minimum(self, a, b, out=None): return OpTreeNode.build("min", a, b, out=out)

    def add         (self, a, b, out=None, iwl=None): return OpTreeNode.build("add", a, b, out=out, iwl=iwl)
    def subtract    (self, a, b, out=None, iwl=None): return OpTreeNode.build("sub", a, b, out=out, iwl=iwl)
    def multiply    (self, a, b, out=None, iwl=None): return OpTreeNode.build("mul", a, b, out=out, iwl=iwl)
    def divide      (self, a, b, out=None, iwl=None): return OpTreeNode.build("div", a, b, out=out, iwl=iwl)
    def true_divide (self, a, b, out=None, iwl=None): return OpTreeNode.build("div", a, b, out=out, iwl=iwl)
    def power       (self, a, b, out=None, iwl=None): return OpTreeNode.build("pow", a, b, out=out, iwl=iwl)
    def reciprocal  (self, a,    out=None, iwl=None): return OpTreeNode.build("div", 1, a, out=out, iwl=iwl)

    def negative    (self, a, out=None): return OpTreeNode.build("neg", a, None, out=out)
    def absolute    (self, a, out=None): return OpTreeNode.build("abs", a, None, out=out)
    def fabs        (self, a, out=None): return OpTreeNode.build("abs", a, None, out=out)

    def sqrt (self, a, out=None, iwl=None): return OpTreeNode.build("sqrt", a, None, out=out, iwl=iwl)
    def sqare(self, a, out=None, iwl=None): return OpTreeNode.build("sqr",  a, None, out=out, iwl=iwl)
    def exp  (self, a, out=None, iwl=None): return OpTreeNode.build("exp",  a, None, out=out, iwl=iwl)
    def exp2 (self, a, out=None, iwl=None): return OpTreeNode.build("exp2", a, None, out=out, iwl=iwl)
    def log  (self, a, out=None, iwl=None): return OpTreeNode.build("log",  a, None, out=out, iwl=iwl)
    def log2 (self, a, out=None, iwl=None): return OpTreeNode.build("log2", a, None, out=out, iwl=iwl)
    def sig  (self, a, out=None, iwl=None): return OpTreeNode.build("sig",  a, None, out=out, iwl=iwl)
    def sig2 (self, a, out=None, iwl=None): return OpTreeNode.build("sig2", a, None, out=out, iwl=iwl)
    def tanh (self, a, out=None, iwl=None): return OpTreeNode.build("tanh", a, None, out=out, iwl=iwl)
    def tanh2(self, a, out=None, iwl=None): return OpTreeNode.build("tanh2",a, None, out=out, iwl=iwl)

    def equal         (self, a, b, out=None): return OpTreeNode.build("eq", a, b, out=out)
    def not_equal     (self, a, b, out=None): return OpTreeNode.build("ne", a, b, out=out)
    def less          (self, a, b, out=None): return OpTreeNode.build("lt", a, b, out=out)
    def less_equal    (self, a, b, out=None): return OpTreeNode.build("le", a, b, out=out)
    def greater       (self, a, b, out=None): return OpTreeNode.build("gt", a, b, out=out)
    def greater_equal (self, a, b, out=None): return OpTreeNode.build("ge", a, b, out=out)

    #TODO: reshape to more efficient dimensions if needed
    def sum(self, a, out=None, iwl=None, axis=None, partial=None):
        return OpTreeNode.build("sum", a, None, out=out, iwl=iwl, axis=axis, partial=partial)

    def max(self, a, out=None, iwl=None, axis=None, partial=None):
        return OpTreeNode.build("max", a, None, out=out, iwl=iwl, axis=axis, partial=partial)

    def min(self, a, out=None, iwl=None, axis=None, partial=None):
        return OpTreeNode.build("min", a, None, out=out, iwl=iwl, axis=axis, partial=partial)

    @staticmethod
    def flex_from_native(value, iwl=None):

        if type(value) not in (int, float, np.ndarray): 
            raise TypeError("Unsupported type: %s" % type(value))

        # find an appropriate iwl and convert
        if iwl is None:

            # find the absolute max and special case all zeros
            if type(value) is np.ndarray:
                max = np.max(np.absolute(value))
                if max == 0: 
                    return (value.astype(np.int16),0)
            else:
                if value == 0: 
                    return (0,0)
                max = value

            # convert the maximum to float and extract the exponent
            iwl = ((struct.unpack('I',struct.pack('f',float(max)))[0] & 0x7f800000) >> 23)-126

            if type(value) is np.ndarray:
                return ((value * 2.0 ** (15 - iwl)).astype(np.int16), iwl)

            return (int(value * 2 ** (15 - iwl)), iwl)

        if type(iwl) is not int:
            raise TypeError("iwl must be an int")

        # convert from int
        if type(value) is int:
            if iwl < 1 or iwl > 15:
                raise ValueError("value(%d) does not fit in flex with requested iwl(%d)" % (value, iwl))
            return value << (15 - iwl)

        #convert from float
        if type(value) is float:
            return int(value * 2 ** (15 - iwl))

        #ndarray
        return (value * 2.0 ** (15 - iwl)).astype(np.int16)

    @staticmethod
    def native_from_flex(flex, iwl):

        #TODO: this needs more work for other types
        return flex * 2 ** (iwl - 15)
        


# For constructing an op tree used in lazy evaluation
class OpTreeNode(tuple):

    def __new__(cls, *args): 

        return tuple.__new__(cls, args)

    @staticmethod
    def build(op, a, b, out=None, **kwargs):

        for arg in (a,b):
            if not isinstance(arg, (int, float, FlexptArray, OpTreeNode, type(None))):
                return NotImplemented

        op_dict = { "op" : op }
        op_dict.update(kwargs)

        node = OpTreeNode(op_dict, a, b)

        # delay execution until assignment
        if out is None: 
            return node

        # passing in an out value counts as assignment
        return node.execute(out=out)

    def execute(self, out):

        stack = self.traverse(list())

        return flexpt_ew.call_compound_ew_kernel(out, *stack)

    # post order walk op tree and produce postfix stack
    def traverse(self, stack):

        # Left
        if type(self[1]) is OpTreeNode:
            self[1].traverse(stack)
        else:
            stack.append(self[1])
        
        # Right
        if type(self[2]) is OpTreeNode:
            self[2].traverse(stack)
        elif self[2] is not None:
            stack.append(self[2])

        stack.append(self[0])

        return stack

    def __add__      (self, other): return self.build("add", self, other)
    def __sub__      (self, other): return self.build("sub", self, other)
    def __mul__      (self, other): return self.build("mul", self, other)
    def __div__      (self, other): return self.build("div", self, other)
    def __truediv__  (self, other): return self.build("div", self, other)
    def __pow__      (self, other): return self.build("pow", self, other)
    def __radd__     (self, other): return self.build("add", other, self)
    def __rsub__     (self, other): return self.build("sub", other, self)
    def __rmul__     (self, other): return self.build("mul", other, self)
    def __rdiv__     (self, other): return self.build("div", other, self)
    def __rtruediv__ (self, other): return self.build("div", other, self)
    def __rpow__     (self, other): return self.build("pow", other, self)
    def __eq__       (self, other): return self.build("eq",  self, other)
    def __ne__       (self, other): return self.build("ne",  self, other)
    def __lt__       (self, other): return self.build("lt",  self, other)
    def __le__       (self, other): return self.build("le",  self, other)
    def __gt__       (self, other): return self.build("gt",  self, other)
    def __ge__       (self, other): return self.build("ge",  self, other)
    def __abs__      (self):        return self.build("abs", self,  None)
    def __neg__      (self):        return self.build("neg", self,  None)

# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# from:
#   Henry S. Warren. Hacker’s Delight. Addison-Wesley Professional, 2002.
#
def magic32(nmax, d):
    nc = ((nmax + 1)//d)*d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2*nbits + 1):
        if 2**p > nc*(d - 1 - (2**p - 1)%d):
            m = (2**p + d - 1 - (2**p - 1)%d)//d
            return (m, p)
    raise ValueError("Can't find magic number for division")

# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 64 bits and the shift
# lops off the lower 32 bits
# from:
#   Henry S. Warren. Hacker’s Delight. Addison-Wesley Professional, 2002.
#
def magic64(d):
    # 3 is a special case that only ends up in the high bits
    # if the maxn is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    
    maxn = 0xffffffff if d == 3 else 0x7fffffff

    magic, shift = magic32(maxn, d)

    if magic != 1: 
        shift -= 32
    
    return (magic, shift)

# flatten a nested list of lists or values
def flatten(lst):
    return sum( ([x] if not isinstance(x, (list,tuple)) else flatten(x) for x in lst), [] )

