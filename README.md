# awesome-gemm [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![awesome-gemm](./img/awesome-gemm.PNG)

> Introduction: This repository is dedicated to compiling an extensive list of frameworks, libraries, and software for matrix-matrix multiplication (A * B = C) optimization. It serves as a comprehensive resource for developers and researchers interested in high-performance computing, numerical analysis, and optimization of matrix operations.

## Table of Contents
- [awesome-gemm ](#awesome-gemm-)
  - [Table of Contents](#table-of-contents)
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

## General Optimization Techniques
- [How To Optimize Gemm](https://github.com/flame/how-to-optimize-gemm): A guide and tutorial on optimizing GEMM operations.
- [GEMM: From Pure C to SSE Optimized Micro Kernels](https://www.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html): An in-depth look into optimizing GEMM from basic C to SSE.

## Frameworks and Development Tools
- [BLIS](https://github.com/flame/blis): A software framework for instantiating high-performance BLAS-like dense linear algebra libraries.
  - Created by [SHPC at UT Austin (formerly FLAME)](https://shpc.oden.utexas.edu/).
- [BLISlab](https://github.com/flame/blislab): A framework for experimenting with and learning about BLIS-like GEMM algorithms.
- [Tensile](https://github.com/ROCm/Tensile): AMD ROCm's library for JIT compiling kernels for matrix multiplications and tensor contractions.

## Libraries

### CPU Libraries
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS): An optimized BLAS library based on GotoBLAS2.
  - Created by [Xianyi Zhang](https://xianyi.github.io/).
- [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html): Intel's Math Kernel Library for optimized mathematical operations.
- [FBGEMM](https://github.com/pytorch/FBGEMM): Facebook's CPU GEMM library optimized for server-side inference.
- [Google gemmlowp](https://github.com/google/gemmlowp): A small self-contained low-precision GEMM library.
- [libFLAME](https://shpc.oden.utexas.edu/libFLAME.html): A high-performance dense linear algebra library.
- [blis_apple](https://github.com/xrq-phys/blis_apple): A BLIS library optimized for Apple M1.

### GPU Libraries
- [NVIDIA CUTLASS 3.3](https://github.com/NVIDIA/cutlass): NVIDIA's template library for CUDA GEMM kernels.
- [NVIDIA cuBLAS](https://developer.nvidia.com/cublas): NVIDIA's implementation of BLAS for CUDA.
- [NVIDIA cuSPARSE](https://developer.nvidia.com/cusparse): NVIDIA's library for sparse matrix operations.
- [hipBLAS](https://github.com/ROCm/hipBLAS): ROCm's BLAS implementation for GPU platforms.
- [hipBLASLt](https://github.com/ROCm/hipBLASLt): Lightweight BLAS implementation for ROCm.
- [hipBLAS-common](https://github.com/ROCm/hipBLAS-common): Common utilities for hipBLAS implementations.
- [OpenAI GEMM](https://github.com/openai/openai-gemm): OpenAI's optimized GEMM implementations.
- [Grouped GEMM](https://github.com/tgale96/grouped_gemm): Efficient implementation of grouped GEMM operations.
- [CoralGemm](https://github.com/AMD-HPC/CoralGemm): AMD's high-performance GEMM implementation.
- [cutlass_fpA_intB_gemm](https://github.com/tlc-pack/cutlass_fpA_intB_gemm): GEMM kernel for fp16 activation and quantized weight.
- [DGEMM on Int8 Tensor Core](https://github.com/enp1s0/ozIMMU): Library intercepting cuBLAS DGEMM function calls.
- [chgemm](https://github.com/tpoisonooo/chgemm): An int8 GEMM project.

### Cross-Platform Libraries
- [MAGMA](https://icl.utk.edu/magma/): Matrix Algebra on GPU and Multicore Architectures.
- [LAPACK](https://www.netlib.org/lapack/): Software library for numerical linear algebra.
- [ARM Compute Library](https://github.com/ARM-software/ComputeLibrary): Machine learning functions optimized for ARM architectures.
- [ViennaCL](https://viennacl.sourceforge.net/): Free open-source linear algebra library for many-core architectures.
- [CUSP](https://github.com/cusplibrary/cusplibrary): A C++ Templated Sparse Matrix Library.
- [CUV](https://github.com/deeplearningais/CUV): A C++ template and Python library for CUDA.

### Language-Specific Libraries
- [NumPy](https://numpy.org/): Python library for scientific computing.
- [SciPy](https://www.scipy.org/): Python library for scientific computing.
- [TensorFlow](https://www.tensorflow.org/): Open-source software library for machine learning.
- [PyTorch](https://pytorch.org/): Open-source software library for machine learning.
- [GemmKernels.jl](https://github.com/JuliaGPU/GemmKernels.jl): Julia package for GEMM operations on GPUs.
- [BLIS.jl](https://github.com/JuliaLinearAlgebra/BLIS.jl): Julia wrapper for BLIS interface.
- [Eigen](https://eigen.tuxfamily.org/): C++ template library for linear algebra.
- [Blaze](https://bitbucket.org/blaze-lib/blaze/src/master/): High-performance C++ math library.
- [Armadillo](https://arma.sourceforge.net/): C++ linear algebra library.
- [Boost uBlas](https://www.boost.org/doc/libs/1_59_0/libs/numeric/ublas/doc/): C++ template class library for BLAS functionality.

## Development Software: Debugging and Profiling
- [HPCToolkit](http://hpctoolkit.org/): An integrated suite of tools for program performance measurement and analysis.
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html): A performance analysis tool for various platforms.
- [gprof](https://hpc.llnl.gov/software/development-environment-software/gprof): A performance analysis tool for Unix applications.
- [gprofng](https://sourceware.org/binutils/docs/gprofng.html): Next generation profiling tool.
  - [gprofng-gui](https://savannah.gnu.org/projects/gprofng-gui/)
- [Memcheck (Valgrind)](https://valgrind.org/docs/manual/mc-manual.html): A memory error detector.
- [FPChecker](https://fpchecker.org/): A tool for detecting floating-point accuracy problems.
- [MegPeak](https://github.com/MegEngine/MegPeak): A tool for testing processor peak computation.

## Learning Resources

### University Courses & Tutorials
- [GPU MODE](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA)
- [HLS Tutorial and Deep Learning Accelerator Design Lab1](https://courses.cs.washington.edu/courses/cse599s/18sp/hw/1.html)
- [UCSB: CS 240A: Applied Parallel Computing](https://sites.cs.ucsb.edu/~tyang/class/240a17/refer.html)
- [UC Berkeley: CS267](https://sites.google.com/lbl.gov/cs267-spr2023)
- [UT Austin: EE382 System-on-Chip (SoC) Design](https://users.ece.utexas.edu/~gerstl/ee382m_f18/labs/lab2.htm)
- [UT Austin (Flame): LAFF-On Programming for High Performance](https://www.cs.utexas.edu/users/flame/laff/pfhp/index.html)

### Selected Papers
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://dl.acm.org/doi/10.1145/2764454). FG Van Zee, RA Van De Geijn. 2015.
- [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://ieeexplore.ieee.org/document/6877334). TM Smith, R Van De Geijn, M Smelyanskiy, JR Hammond, FG Van Zee. 2014.
- [Model-driven Level 3 BLAS Performance Optimization on Loongson 3A Processor](https://ieeexplore.ieee.org/document/6413635). Z Xianyi, W Qian, Z Yunquan. 2012.
- [High-performance implementation of the level-3 BLAS](https://dl.acm.org/doi/10.1145/1377603.1377607). K Goto, R Van De Geijn. 2008.
- [Anatomy of high-performance matrix multiplication](https://dl.acm.org/doi/10.1145/1356052.1356053). K Goto, RA Geijn. 2008.

### Lecture Notes
- [ORNL: CUDA C++ Exercise: Basic Linear Algebra Kernels: GEMM Optimization Strategies](https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final)
- [Stanford: BLAS-level CPU Performance in 100 Lines of C](https://cs.stanford.edu/people/shadjis/blas.html)
- [Puedue: Optimizing matrix multiplication](https://www.cs.purdue.edu/homes/grr/cs250/lab6-cache/optimizingMatrixMultiplication.pdf)
- [NJIT: Optimize Matrix Multiplication](https://web.njit.edu/~apv6/courses/hw1.html)

### Blogs
- [Optimizing Matrix Multiplication](https://coffeebeforearch.github.io/2020/06/23/mmul.html)
- [GEMM caching](https://zhuanlan.zhihu.com/p/69700540)
- [Matrix Multiplication on CPU](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Optimizing matrix multiplication: cache + OpenMP](https://www.mgaillard.fr/2020/08/29/matrix-multiplication-optimizing.html)
- [Tuning matrix multiplication (GEMM) for Intel GPUs](https://www.ibiblio.org/e-notes/webgl/gpu/mul/intel.htm)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Building a FAST matrix multiplication algorithm](https://v0dro.in/blog/2018/05/01/building-a-fast-matrix-multiplication-algorithm/)
- [Matrix-Matrix Product Experiments with BLAZE](https://www.mathematik.uni-ulm.de/~lehn/test_blaze/index.html)
- [CUDA Learn Notes](https://github.com/DefTruth/CUDA-Learn-Notes): Comprehensive notes on CUDA programming and optimization.
- [CUDA GEMM Optimization](https://github.com/leimao/CUDA-GEMM-Optimization): Step-by-step GEMM optimization tutorial.
- [The OpenBLAS Project and Matrix Multiplication Optimization](https://www.leiphone.com/category/yanxishe/Puevv3ZWxn0heoEv.html) (Chinese)
- [Step by step optimization of cuda sgemm](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) (Chinese)
- [OpenBLAS gemm from scratch](https://zhuanlan.zhihu.com/p/65436463) (Chinese)
- [The Proper Approach to CUDA for Beginners: How to Optimize GEMM](https://zhuanlan.zhihu.com/p/478846788) (Chinese)
- [ARMv7 4x4kernel Optimization Practice](https://zhuanlan.zhihu.com/p/333799799) (Chinese)

### Other Resources
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/): New cuBLAS 12.0 Features and Matrix Multiplication Performance.
- [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html): Guide to matrix multiplication on NVIDIA GPUs.
- [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html): Programming language for efficient GPU code.
- [perf-book](https://github.com/dendibakh/perf-book): "Performance Analysis and Tuning on Modern CPU" by Denis Bakhvalov.

## Example Implementations
- [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA): Step-by-step optimization of matrix multiplication in CUDA.
- [simple-gemm](https://github.com/williamfgc/simple-gemm): Collection of simple GEMM implementations.
- [YHs_Sample](https://github.com/Yinghan-Li/YHs_Sample): A CUDA implementation of GEMM.
- [how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm): A row-major matmul optimization tutorial.
- [GEMM](https://github.com/iVishalr/GEMM): Fast Matrix Multiplication Implementation in C.

