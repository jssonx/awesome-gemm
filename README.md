# awesome-gemm [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![awesome-gemm](./img/awesome-gemm.PNG)

> **Introduction**: This repository is dedicated to compiling an extensive list of frameworks, libraries, and software for matrix-matrix multiplication (A * B = C) optimization. It serves as a comprehensive resource for developers and researchers interested in high-performance computing, numerical analysis, and optimization of matrix operations.

## Table of Contents
- [Fundamental Theories and Concepts](#fundamental-theories-and-concepts)
- [General Optimization Techniques](#general-optimization-techniques)
- [Frameworks and Development Tools](#frameworks-and-development-tools)
- [Libraries](#libraries)
  - [CPU Libraries](#cpu-libraries)
  - [GPU Libraries](#gpu-libraries)
  - [Cross-Platform Libraries](#cross-platform-libraries)
  - [Language-Specific Libraries](#language-specific-libraries)
- [Development Software: Debugging and Profiling](#development-software-debugging-and-profiling)
- [Learning Resources](#learning-resources)
  - [University Courses \& Tutorials](#university-courses--tutorials)
  - [Selected Papers](#selected-papers)
  - [Lecture Notes](#lecture-notes)
  - [Blogs](#blogs)
  - [Other Resources](#other-resources)
- [Example Implementations](#example-implementations)

## Fundamental Theories and Concepts
- [General Matrix Multiply (GeMM)](https://spatial-lang.org/gemm)
- [General Matrix Multiply (Intel)](https://www.intel.com/content/dam/develop/external/us/en/documents/intel-ocl-gemm.pdf)
- [Strassen's Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm): An algorithm for matrix multiplication that is faster than the conventional algorithm for large matrices.
- [Winograd's Algorithm](https://en.wikipedia.org/wiki/Winograd_algorithm): An efficient algorithm for matrix multiplication that reduces the number of multiplications.

## General Optimization Techniques
- [How To Optimize Gemm](https://github.com/flame/how-to-optimize-gemm): A guide and tutorial on optimizing GEMM operations.
- [GEMM: From Pure C to SSE Optimized Micro Kernels](https://www.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html): An in-depth look into optimizing GEMM from basic C to SSE.

## Frameworks and Development Tools
- [BLIS](https://github.com/flame/blis): A software framework for instantiating high-performance BLAS-like dense linear algebra libraries. [`BSD-3-Clause`](https://github.com/flame/blis/blob/master/LICENSE.md)
  - Created by [SHPC at UT Austin (formerly FLAME)](https://shpc.oden.utexas.edu/).
- [BLISlab](https://github.com/flame/blislab): A framework for experimenting with and learning about BLIS-like GEMM algorithms.
- [Tensile](https://github.com/ROCm/Tensile): AMD ROCm's library for JIT compiling kernels for matrix multiplications and tensor contractions. [`MIT`](https://github.com/ROCm/Tensile/blob/develop/LICENSE.md)

## Libraries

### CPU Libraries
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS): An optimized BLAS library based on GotoBLAS2. [`BSD-3-Clause`](https://github.com/xianyi/OpenBLAS/blob/develop/LICENSE)
  - Created by [Xianyi Zhang](https://xianyi.github.io/).
- [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html): Intel's Math Kernel Library for optimized mathematical operations.
- [oneDNN (formerly MKL-DNN)](https://github.com/oneapi-src/oneDNN): An open-source cross-platform performance library of deep learning building blocks, optimized for Intel architectures. [`Apache-2.0`](https://github.com/oneapi-src/oneDNN/blob/master/LICENSE)
- [FBGEMM](https://github.com/pytorch/FBGEMM): Facebook's CPU GEMM library optimized for server-side inference. [`BSD-3-Clause`](https://github.com/pytorch/FBGEMM/blob/master/LICENSE)
- [Google gemmlowp](https://github.com/google/gemmlowp): A small self-contained low-precision GEMM library. [`Apache-2.0`](https://github.com/google/gemmlowp/blob/master/LICENSE)
- [libFLAME](https://shpc.oden.utexas.edu/libFLAME.html): A high-performance dense linear algebra library. [`BSD-3-Clause`](https://github.com/flame/libflame/blob/master/LICENSE.txt)
- [blis_apple](https://github.com/xrq-phys/blis_apple): A BLIS library optimized for Apple M1. [`BSD-3-Clause`](https://github.com/xrq-phys/blis_apple/blob/amx-dev/LICENSE)
- [BLASFEO](https://github.com/giaf/blasfeo): Basic Linear Algebra Subroutines for Embedded Optimization, tailored for small to medium-sized matrices common in embedded optimization. [`BSD-2-Clause`](https://github.com/giaf/blasfeo/blob/master/LICENSE.txt)
- [LIBXSMM](https://github.com/hfp/libxsmm): A library targeting small, dense or sparse matrix multiplications, especially useful for small GEMM kernels. [`BSD-3-Clause`](https://github.com/hfp/libxsmm/blob/master/LICENSE.md)

### GPU Libraries
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass): NVIDIA's template library for CUDA GEMM kernels. [`BSD-3-Clause`](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt)
- [NVIDIA cuBLAS](https://developer.nvidia.com/cublas): NVIDIA's implementation of BLAS for CUDA. [`NVIDIA Software License`](https://docs.nvidia.com/cuda/eula/index.html)
- [NVIDIA cuSPARSE](https://developer.nvidia.com/cusparse): NVIDIA's library for sparse matrix operations. [`NVIDIA Software License`](https://docs.nvidia.com/cuda/eula/index.html)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn): NVIDIA's CUDA Deep Neural Network library, providing optimized primitives for deep learning, including matrix multiplication. [`NVIDIA Software License`](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html)
- [hipBLAS](https://github.com/ROCm/hipBLAS): ROCm's BLAS implementation for GPU platforms. [`MIT`](https://github.com/ROCm/hipBLAS/blob/develop/LICENSE.md)
- [hipBLASLt](https://github.com/ROCm/hipBLASLt): Lightweight BLAS implementation for ROCm. [`MIT`](https://github.com/ROCm/hipBLASLt/blob/develop/LICENSE.md)
- [hipBLAS-common](https://github.com/ROCm/hipBLAS-common): Common utilities for hipBLAS implementations. [`MIT`](https://github.com/ROCm/hipBLAS-common/blob/develop/LICENSE.md)
- [OpenAI GEMM](https://github.com/openai/openai-gemm): OpenAI's optimized GEMM implementations. [`MIT`](https://github.com/openai/openai-gemm/blob/master/LICENSE)
- [Grouped GEMM](https://github.com/tgale96/grouped_gemm): Efficient implementation of grouped GEMM operations. [`Apache-2.0`](https://github.com/tgale96/grouped_gemm/blob/main/LICENSE)
- [CoralGemm](https://github.com/AMD-HPC/CoralGemm): AMD's high-performance GEMM implementation. [`MIT`](https://github.com/AMD-HPC/CoralGemm/blob/master/LICENSE.md)
- [cutlass_fpA_intB_gemm](https://github.com/tlc-pack/cutlass_fpA_intB_gemm): GEMM kernel for fp16 activation and quantized weight. [`Apache-2.0`](https://github.com/tlc-pack/cutlass_fpA_intB_gemm/blob/main/LICENSE)
- [DGEMM on Int8 Tensor Core](https://github.com/enp1s0/ozIMMU): Library intercepting cuBLAS DGEMM function calls. [`MIT`](https://github.com/enp1s0/ozIMMU/blob/main/LICENSE)
- [chgemm](https://github.com/tpoisonooo/chgemm): An int8 GEMM project.
- [clBLAS](https://github.com/clMathLibraries/clBLAS): A software library containing BLAS functions written in OpenCL, making it portable across different GPU vendors. [`Apache-2.0`](https://github.com/clMathLibraries/clBLAS/blob/master/LICENSE)
- [clBLAST](https://github.com/CNugteren/CLBlast): An optimized OpenCL BLAS library tuned for performance. [`Apache-2.0`](https://github.com/CNugteren/CLBlast/blob/master/LICENSE)
- [ArrayFire](https://github.com/arrayfire/arrayfire): A general-purpose GPU library that simplifies GPU computing with high-level functions, including matrix operations. [`BSD-3-Clause`](https://github.com/arrayfire/arrayfire/blob/master/LICENSE)

### Cross-Platform Libraries
- [MAGMA](https://github.com/icl-utk-edu/magma): Matrix Algebra on GPU and Multicore Architectures. [`BSD-3-Clause`](https://github.com/icl-utk-edu/magma/blob/master/COPYRIGHT)
- [LAPACK](https://www.netlib.org/lapack/): Software library for numerical linear algebra. [`BSD-3-Clause`](https://www.netlib.org/lapack/LICENSE.txt)
- [ARM Compute Library](https://github.com/ARM-software/ComputeLibrary): Machine learning functions optimized for ARM architectures. [`MIT`](https://github.com/ARM-software/ComputeLibrary/blob/main/LICENSES/MIT.txt) [`Apache-2.0`](https://github.com/ARM-software/ComputeLibrary/blob/main/LICENSES/Apache-2.0.txt)
- [viennacl-dev](https://github.com/viennacl/viennacl-dev): Free open-source linear algebra library for many-core architectures. [`MIT`](https://github.com/viennacl/viennacl-dev/blob/master/LICENSE)
- [CUSP](https://github.com/cusplibrary/cusplibrary): A C++ Templated Sparse Matrix Library. [`Apache-2.0`](https://github.com/cusplibrary/cusplibrary/blob/master/LICENSE)
- [CUV](https://github.com/deeplearningais/CUV): A C++ template and Python library for CUDA.
- [Ginkgo](https://github.com/ginkgo-project/ginkgo): A high-performance linear algebra library for many-core systems, designed for flexibility and efficiency. [`BSD-3-Clause`](https://github.com/ginkgo-project/ginkgo/blob/develop/LICENSE)

### Language-Specific Libraries
- [NumPy](https://github.com/numpy/numpy): Python library for scientific computing. [`BSD-3-Clause`](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [SciPy](https://github.com/scipy/scipy): Python library for scientific computing. [`BSD-3-Clause`](https://github.com/scipy/scipy/blob/main/LICENSE.txt)
- [TensorFlow](https://github.com/tensorflow/tensorflow): Open-source software library for machine learning. [`Apache-2.0`](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [TensorFlow XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla): A domain-specific compiler for linear algebra that optimizes TensorFlow computations. [`Apache-2.0`](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
- [JAX](https://github.com/google/jax): A Python library for high-performance machine learning research, enabling transformations of numerical functions. [`Apache-2.0`](https://github.com/google/jax/blob/main/LICENSE)
- [PyTorch](https://github.com/pytorch/pytorch): Open-source software library for machine learning. [`BSD-3-Clause`](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [GemmKernels.jl](https://github.com/JuliaGPU/GemmKernels.jl): Julia package for GEMM operations on GPUs. [`BSD-3-Clause`](https://github.com/JuliaGPU/GemmKernels.jl/blob/master/LICENSE)
- [BLIS.jl](https://github.com/JuliaLinearAlgebra/BLIS.jl): Julia wrapper for BLIS interface. [`BSD-3-Clause`](https://github.com/JuliaLinearAlgebra/BLIS.jl/blob/master/LICENSE)
- [Eigen](https://gitlab.com/libeigen/eigen): C++ template library for linear algebra. [`MPL2`](https://gitlab.com/libeigen/eigen/-/blob/master/COPYING.MPL2)
- [Blaze](https://bitbucket.org/blaze-lib/blaze/src/master/): High-performance C++ math library. [`BSD-3-Clause`](https://bitbucket.org/blaze-lib/blaze/src/master/LICENSE)
- [Armadillo](https://arma.sourceforge.net/): C++ linear algebra library.
- [Boost uBlas](https://www.boost.org/doc/libs/1_59_0/libs/numeric/ublas/doc/): C++ template class library for BLAS functionality. [`Boost Software License 1.0`](https://www.boost.org/LICENSE_1_0.txt)

## Development Software: Debugging and Profiling
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html): A performance analysis tool for various platforms, ideal for profiling and optimizing applications on Intel architectures.
- [Intel Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/advisor.html): A tool for vectorization optimization and memory layout transformations to improve application performance.
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems): A system-wide performance analysis tool designed to visualize application algorithms, optimize performance, and enhance efficiency on NVIDIA GPUs. [`NVIDIA SOFTWARE LICENSE AGREEMENT`](https://docs.nvidia.com/nsight-systems/CopyrightAndLicenses/index.html)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute): A performance analysis tool for CUDA kernels, providing detailed performance metrics and API debugging.
- [Nsight Visual Studio Edition](https://developer.nvidia.com/nsight-visual-studio-edition): An integrated development environment for debugging and profiling CUDA applications within Visual Studio.
- [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview): NVIDIA's command-line profiler for CUDA applications. [`NVIDIA End User License Agreement`](https://docs.nvidia.com/cuda/eula/index.html)
- [ROCm Profiler](https://github.com/ROCm/rocprofiler): AMD's performance analysis tool for profiling applications running on ROCm platforms. [`MIT`](https://github.com/ROCm/rocprofiler/blob/amd-staging/LICENSE)
- [HPCToolkit](https://gitlab.com/hpctoolkit/hpctoolkit): An integrated suite of tools for program performance measurement and analysis across a range of architectures. [`BSD-3-Clause`](https://gitlab.com/hpctoolkit/hpctoolkit/-/blob/develop/LICENSE.md)
- [TAU (Tuning and Analysis Utilities)](https://www.cs.uoregon.edu/research/tau/home.php): A performance evaluation tool framework for high-performance parallel programs.
- [Perf](https://perf.wiki.kernel.org/index.php/Main_Page): A performance analyzing tool in Linux, useful for profiling CPU performance counters and system-level metrics. [`GPLv2`](https://github.com/torvalds/linux/blob/master/COPYING)
- [gprof](https://sourceware.org/binutils/docs/gprof/): A performance analysis tool for Unix applications, useful for identifying program bottlenecks. [`GPLv3`](https://www.gnu.org/licenses/gpl-3.0.html)
- [gprofng](https://sourceware.org/binutils/docs/gprofng.html): The next-generation GNU profiling tool with improved capabilities. [`GPLv3`](https://www.gnu.org/licenses/gpl-3.0.html)
  - [gprofng-gui](https://savannah.gnu.org/projects/gprofng-gui/): A graphical user interface for gprofng. [`GPLv3`](https://www.gnu.org/licenses/gpl-3.0.html)
- [LIKWID](https://github.com/RRZE-HPC/likwid): A suite of command-line tools for performance-oriented programmers to profile and optimize their applications. [`GPLv3`](https://github.com/RRZE-HPC/likwid/blob/master/COPYING)
- [VAMPIR](https://vampir.eu/): A tool suite for performance analysis and visualization of parallel programs, aiding in identifying performance issues. [`Proprietary`](https://vampir.eu/)
- [Extrae](https://tools.bsc.es/extrae): A package that generates trace files for performance analysis, which can be visualized with Paraver. [`GPLv2.1`](https://github.com/bsc-performance-tools/extrae/blob/master/COPYING)
- [Memcheck (Valgrind)](https://valgrind.org/docs/manual/mc-manual.html): A memory error detector that helps identify issues like memory leaks and invalid memory access. [`GPLv2`](https://sourceware.org/git/?p=valgrind.git;a=blob_plain;f=COPYING;hb=HEAD)
- [FPChecker](https://github.com/LLNL/FPChecker): A tool for detecting floating-point accuracy problems in applications. [`BSD-3-Clause`](https://github.com/LLNL/FPChecker/blob/master/LICENSE)
- [MegPeak](https://github.com/MegEngine/MegPeak): A tool for testing processor peak computation performance, useful for benchmarking. [`Apache-2.0`](https://github.com/MegEngine/MegPeak/blob/main/LICENSE)

## Learning Resources

### University Courses & Tutorials
- [GPU MODE](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA)
- [HLS Tutorial and Deep Learning Accelerator Design Lab1](https://courses.cs.washington.edu/courses/cse599s/18sp/hw/1.html)
- [UCSB: CS 240A: Applied Parallel Computing](https://sites.cs.ucsb.edu/~tyang/class/240a17/refer.html)
- [UC Berkeley: CS267](https://sites.google.com/lbl.gov/cs267-spr2023)
- [UT Austin: EE382 System-on-Chip (SoC) Design](https://users.ece.utexas.edu/~gerstl/ee382m_f18/labs/lab2.htm)
- [UT Austin (Flame): LAFF-On Programming for High Performance](https://www.cs.utexas.edu/users/flame/laff/pfhp/index.html)
- [MIT OpenCourseWare: Performance Engineering of Software Systems](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/): Techniques for writing fast code, including optimization of matrix operations.

### Selected Papers
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://dl.acm.org/doi/10.1145/2764454). FG Van Zee, RA Van De Geijn. 2015.
- [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://ieeexplore.ieee.org/document/6877334). TM Smith, R Van De Geijn, M Smelyanskiy, JR Hammond, FG Van Zee. 2014.
- [Model-driven Level 3 BLAS Performance Optimization on Loongson 3A Processor](https://ieeexplore.ieee.org/document/6413635). Z Xianyi, W Qian, Z Yunquan. 2012.
- [High-performance implementation of the level-3 BLAS](https://dl.acm.org/doi/10.1145/1377603.1377607). K Goto, R Van De Geijn. 2008.
- [Anatomy of high-performance matrix multiplication](https://dl.acm.org/doi/10.1145/1356052.1356053). K Goto, RA Geijn. 2008.

### Lecture Notes
- [ORNL: CUDA C++ Exercise: Basic Linear Algebra Kernels: GEMM Optimization Strategies](https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final)
- [Stanford: BLAS-level CPU Performance in 100 Lines of C](https://cs.stanford.edu/people/shadjis/blas.html)
- [Purdue: Optimizing Matrix Multiplication](https://www.cs.purdue.edu/homes/grr/cs250/lab6-cache/optimizingMatrixMultiplication.pdf)
- [NJIT: Optimize Matrix Multiplication](https://web.njit.edu/~apv6/courses/hw1.html)
- [Optimizing Matrix Multiplication using SIMD and Parallelization](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/6-172-fall-2018/lecture-notes/MIT6_172F18_lec5.pdf)

### Blogs
- [Distributed GEMM - A novel CUTLASS-based implementation of Tensor Parallelism for NVLink-enabled systems](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b)
- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
- [CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: A Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Matrix Multiplication on CPU](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Optimizing Matrix Multiplication](https://coffeebeforearch.github.io/2020/06/23/mmul.html)
- [Optimizing Matrix Multiplication: Cache + OpenMP](https://www.mgaillard.fr/2020/08/29/matrix-multiplication-optimizing.html)
- [Tuning Matrix Multiplication (GEMM) for Intel GPUs](https://www.ibiblio.org/e-notes/webgl/gpu/mul/intel.htm)
- [Building a FAST Matrix Multiplication Algorithm](https://v0dro.in/blog/2018/05/01/building-a-fast-matrix-multiplication-algorithm/)
- [Matrix-Matrix Product Experiments with BLAZE](https://www.mathematik.uni-ulm.de/~lehn/test_blaze/index.html)
- [CUDA Learn Notes](https://github.com/DefTruth/CUDA-Learn-Notes)
- [CUDA GEMM Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)
- [Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)
- [The OpenBLAS Project and Matrix Multiplication Optimization](https://www.leiphone.com/category/yanxishe/Puevv3ZWxn0heoEv.html) (Chinese)
- [Step by Step Optimization of CUDA SGEMM](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) (Chinese)
- [OpenBLAS GEMM from Scratch](https://zhuanlan.zhihu.com/p/65436463) (Chinese)
- [The Proper Approach to CUDA for Beginners: How to Optimize GEMM](https://zhuanlan.zhihu.com/p/478846788) (Chinese)
- [ARMv7 4x4kernel Optimization Practice](https://zhuanlan.zhihu.com/p/333799799) (Chinese)
- [GEMM Caching](https://zhuanlan.zhihu.com/p/69700540) (Chinese)

### Other Resources
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/): New cuBLAS 12.0 Features and Matrix Multiplication Performance.
- [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html): Guide to matrix multiplication on NVIDIA GPUs.
- [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html): Programming language for efficient GPU code.
- [perf-book](https://github.com/dendibakh/perf-book): "Performance Analysis and Tuning on Modern CPU" by Denis Bakhvalov.
- [The High-Performance Computing (HPC) Garage](https://github.com/hpcgarage): A collection of HPC codes and tools from the Innovative Computing Laboratory (ICL) at the University of Tennessee.

## Example Implementations
- [Toy HGEMM Library using Tensor Cores with MMA/WMMA/CuTe](https://github.com/DefTruth/hgemm-tensorcores-mma): May achieve the 98%~100% performance of cuBLAS. [`GPLv3`](https://github.com/DefTruth/hgemm-tensorcores-mma/blob/main/LICENSE)
- [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA): Step-by-step optimization of matrix multiplication in CUDA. [`MIT`](https://github.com/siboehm/SGEMM_CUDA/blob/master/LICENSE)
- [simple-gemm](https://github.com/williamfgc/simple-gemm): Collection of simple GEMM implementations. [`MIT`](https://github.com/williamfgc/simple-gemm/blob/main/LICENSE)
- [YHs_Sample](https://github.com/Yinghan-Li/YHs_Sample): A CUDA implementation of GEMM. [`GPLv3`](https://github.com/Yinghan-Li/YHs_Sample/blob/master/LICENSE)
- [how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm): A row-major matmul optimization tutorial. [`GPLv3`](https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/LICENSE)
- [GEMM](https://github.com/iVishalr/GEMM): Fast Matrix Multiplication Implementation in C. [`MIT`](https://github.com/iVishalr/GEMM/blob/main/LICENSE)
- [GEMM Optimization with LIBXSMM](https://github.com/hfp/libxsmm/tree/main/samples): Sample codes showing how to use LIBXSMM for optimizing small matrix multiplications. [`BSD-3-Clause`](https://github.com/libxsmm/libxsmm/blob/main/LICENSE.md)
- [Deep Learning GEMM Benchmarks](https://github.com/baidu-research/DeepBench): Benchmarks for measuring the performance of basic deep learning operations including GEMM. [`Apache-2.0`](https://github.com/baidu-research/DeepBench/blob/master/LICENSE)

---

*This curated list aims to be a comprehensive resource for anyone interested in the optimization of matrix-matrix multiplication. Contributions and suggestions are welcome to help keep this list up-to-date and useful for the community.*