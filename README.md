### Preview release of convolutional kernels

This is a proof-of-concept preview release of the main GPU kernels
used in a convolutional neural network (CNN). They are being
incorporated into a forthcoming release of Nervana's full-featured
Deep Learning Library, which is currently in limited beta. The preview
includes convolutional fprop-backprop-update kernels, dense matrix
multiply (GEMM) kernels, and automatically generated element-wise
kernels. The kernels use an underlying 16-bit representation used in a
recent paper by Courbariaux et al.<sup>[1](#refs)</sup> from the Bengio lab,
where they demonstrate that representations as low as 12 bits are
sufficent for both learning and inference for several state-of-the-art
deep networks.

We are releasing this library to solicit feedback for our full
release. In addition to being very high performance, it facilitates
explorations of limited bit-width, integer-like numerical
representations for deep networks.

#### Features

The kernels were designed using
[MaxAs](https://github.com/NervanaSystems/maxas), our assembler for
the NVIDIA Maxwell GPU.

Here are some features of the kernels:

1. They achieve near full utilization on the Maxwell GPU for all
kernels, including convolutional backprop and update, and are one of
the fastest implementations we know of.

2. The 16-bit representation means twice the memory I/O bandwidth and
twice the amount of data elements that can fit into DDR.

3. The kernels allow you to set your precision for each operand
enabling algorithmic explorations at different bit widths.

4. The convolutional kernel features and function arguments are
identical to those of cuDNN, with the addition of 3-D convolutions.

We have a library of fast kernels we use for our internal
efforts. These kernels support a wider range of problem sizes,
operations, and numerical formats. We plan to share them in a future
release.

#### Prior work

Vanhoucke et al.<sup>[2](#refs)</sup> demonstrated limited-precision
fixed-point SIMD optimizations for CPUs with significant speed
improvements for inference in a mixed HMM/NN large vocabulary
system. Coubariaux et al.<sup>[1](#refs)</sup> recently showed success
with limited precision for both inference and learing.

We modeled our kernels' functionality on NVIDIA's cuDNN library
described in Chetlur et al.<sup>[3](#refs)</sup>. The GEMM kernels are
also based on NVIDIA's work as we previously noted
[here](https://github.com/NervanaSystems/maxas/wiki/SGEMM).

There is a large number of academic results with CNNs and high quality
open source packages<sup>[4-6](#refs)</sup> from which we have
learned.  Note that a Fourier domain approach such as Vasilache et
al.<sup>[5](#refs)</sup> can outperform our kernels for certain
problem sizes at the cost of additional memory and flexibility.

Andrew Lavin very recently demonstrated<sup>[7](#refs)</sup> a similar
approach using [MaxAs](https://github.com/NervanaSystems/maxas) for
performing forward prop in 32-bit floating point with full utilization
on Maxwell ([repository](https://github.com/eBay/maxDNN),
[writeup](http://arxiv.org/abs/1501.06633)). The differences with our
kernels are our 16-bit representation, support for backprop and update
in addition to fprop, in-place zero padding, upscaling, and 3D
convolutions.

#### How to run

The kernels are wrapped using Python classes which have `numpy` and
`pycuda` as dependencies. The wrapper classes borrow from `pycuda's`
GPUArray class. The kernels run on NVIDIA Maxwell only and have been
tested on Ubuntu 14.04.

The following demonstration scripts are included:

- `convolution.py` is a simple script that runs forward prop, backprop
and update kernels.

        % python convolution.py
        fprop_conv  NCK: (128,192,384) DHW:[1, 13, 13]
                    TRS:[1, 3, 3] MPQ:[1, 11, 11]:  4.478526 msecs 4589.5 gflops
        bprop_conv  NCK: (128,192,384) DHW:[1, 13, 13]
                    TRS:[1, 3, 3] MPQ:[1, 11, 11]:  4.429529 msecs 4640.3 gflops
        update_conv NCK: (128,192,384) DHW:[1, 13, 13]
                    TRS:[1, 3, 3] MPQ:[1, 11, 11]:  4.437881 msecs 4631.5 gflops

- `gemm.py` is a simple script that runs several GEMM kernels.

- `bench.py` is a script to generate some of the numbers for Soumith Chintala's
[page](https://github.com/soumith/convnet-benchmarks).

        % python bench.py

        Our numbers (Layers 1-5 and total in msecs, lower is better):

          forward: [  41.  112.   42.    5.    9.]  total=208
          backward: [ 139.  240.   87.    9.   18.]  total=493
          gradInput: [  81.  124.   46.    5.    9.]  total=264
          gradWeight: [  58.  116.   41.    4.    9.]  total=228

        For comparison, cuDNN R2 reference numbers, from
        https://github.com/soumith/convnet-benchmarks (accessed 2/2/15):

          forward: [  90.  218.   79.    9.   17.]  total=413
          backward: [ 189.  606.  230.   27.   47.]  total=1099
          gradInput: [  91.  344.  130.   15.   20.]  total=600
          gradWeight: [  98.  262.  100.   12.   27.]  total=499

The scripts output basic performance metrics. For more detail, you can
use NVIDIA CUDA's
[`nvprof`](http://docs.nvidia.com/cuda/profiler-users-guide/)
tool. Keep in mind there are a number of subtle factors determining
utilization and FLOPS numbers, such as the entropy of the
inputs. Please consult the documentation.

#### License

Please note the terms of the software in LICENSE.txt, including
non-commercial use and reverse engineering clauses. If you wish to
license this or related software, please contact us at
license@nervanasys.com.  Flexpoint&trade; refers to the different
numerical representations we use in our CPU and GPU libraries, as well
as in Nervana's forthcoming hardware.

#### References <a name="refs"></a>

1. Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre
David. [*Low precision arithmetic for deep learning.*](http://arxiv.org/abs/1412.7024)
arXiv preprint arXiv:1412.7024 (2014).

2. Vanhoucke, Vincent, Andrew Senior, and Mark
Z. Mao. [*Improving the speed of neural networks on CPUs.*]
(http://research.google.com/pubs/pub37631.html) Proc. Deep Learning
and Unsupervised Feature Learning NIPS Workshop (2011).

3. Chetlur, Sharan, Cliff Woolley, Philippe Vandermersch, Jonathan
Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer.
[*cuDNN: Efficient primitives for deep learning.*](http://arxiv.org/abs/1410.0759)
arXiv preprint arXiv:1410.0759 (2014).

4. Jia, Yangqing, Evan Shelhamer, Jeff Donahue, Sergey Karayev,
Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor
Darrell. [*Caffe: Convolutional architecture for fast feature embedding.*](http://caffe.berkeleyvision.org/)
In Proceedings of the ACM International Conference on Multimedia,
pp. 675-678. ACM (2014).

5. Vasilache, Nicolas, Jeff Johnson, Michael Mathieu, Soumith
Chintala, Serkan Piantino, and Yann LeCun.
[*Fast Convolutional Nets With fbfft: A GPU Performance Evaluation.*](http://arxiv.org/abs/1412.7580)
arXiv preprint arXiv:1412.7580 (2014).

6. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton.
[*Imagenet classification with deep convolutional neural networks.*](https://code.google.com/p/cuda-convnet2/)
NIPS (2012).

7. Lavin, Andrew.
[*maxDNN: An Efficient Convolution Kernel for Deep Learning with Maxwell GPUs.*](http://arxiv.org/abs/1501.06633)
arXiv preprint arXiv:1501.06633 (2015).
