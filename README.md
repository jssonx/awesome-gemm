# awesome-gemm [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![awesome-gemm](./img/awesome-gemm.PNG)

> Introduction: This repository is dedicated to compiling an extensive list of frameworks, libraries, and software for matrix-matrix multiplication (A * B = C) optimization. It serves as a comprehensive resource for developers and researchers interested in high-performance computing, numerical analysis, and optimization of matrix operations.

# Table of Contents
- [Fundamental Theories and Concepts](#fundamental-theories-and-concepts)
- [General Optimization Techniques](#general-optimization-techniques)
- [Frameworks](#frameworks)
- [Libraries](#libraries)
- [Development Software: Debugging and Profiling](#development-software-debugging-and-profiling)
- [University Courses \& Tutorials](#university-courses--tutorials)
- [Selected Papers](#selected-papers)
- [Lecture Notes](#lecture-notes)
- [Blogs](#blogs)
- [Other Learning Resources](#other-learning-resources)
- [Tiny Examples](#tiny-examples)
- [How to Contribute](#how-to-contribute)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Fundamental Theories and Concepts
- [General Matrix Multiply (GeMM)](https://spatial-lang.org/gemm)
- [General Matrix Multiply (Intel)](https://www.intel.com/content/dam/develop/external/us/en/documents/intel-ocl-gemm.pdf)

## General Optimization Techniques
- [How To Optimize Gemm](https://github.com/flame/how-to-optimize-gemm): A guide and tutorial on optimizing GEMM operations.
- [GEMM: From Pure C to SSE Optimized Micro Kernels](https://www.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html): An in-depth look into optimizing GEMM from basic C to SSE.

## Frameworks
- [BLIS](https://github.com/flame/blis): A software framework for instantiating high-performance BLAS-like dense linear algebra libraries.
  - Created by [SHPC at UT Austin (formerly FLAME)](https://shpc.oden.utexas.edu/).
- [BLISlab](https://github.com/flame/blislab): A framework for experimenting with and learning about BLIS-like GEMM algorithms.

## Libraries
- [gemmlowp: a small self-contained low-precision GEMM library](https://github.com/google/gemmlowp): A compact library for low-precision GEMM optimization by Google.
- [Eigen](https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html): A C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
- [MAGMA (Matrix Algebra on GPU and Multicore Architectures)](https://icl.utk.edu/magma/): A collection of next-generation linear algebra libraries for heterogeneous computing.
- [LAPACK](https://www.netlib.org/lapack/): A software library for numerical linear algebra.
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS?tab=readme-ov-file): An optimized BLAS library based on GotoBLAS2.
  - Created by [Xianyi Zhang](https://xianyi.github.io/).
- [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html): Intel's Math Kernel Library offering highly optimized, threaded, and vectorized functions for mathematical operations.
- [ARM Compute Library](https://github.com/ARM-software/ComputeLibrary): A collection of low-level machine learning functions optimized for Arm® Cortex®-A, Arm® Neoverse® and Arm® Mali™ GPUs architectures.
- [NumPy](https://numpy.org/): A Python library for scientific computing with a focus on array operations.
- [SciPy](https://www.scipy.org/): A Python library for scientific computing with a focus on linear algebra.
- [TensorFlow](https://www.tensorflow.org/): An open-source software library for machine learning.
- [PyTorch](https://pytorch.org/): An open-source software library for machine learning.
- [NVIDIA cuBLAS](https://developer.nvidia.com/cublas): NVIDIA's implementation of the BLAS (Basic Linear Algebra Subprograms) on top of its CUDA runtime.
- [NVIDIA cuSPARSE](https://developer.nvidia.com/cusparse): NVIDIA's library for sparse matrix operations on CUDA.
- [libFLAME](https://shpc.oden.utexas.edu/libFLAME.html): A high performance dense linaer algebra library that is the result of the FLAME methodology for systematically developing dense linear algebra libraries.
- [ViennaCL](https://viennacl.sourceforge.net/): a free open-source linear algebra library for computations on many-core architectures (GPUs, MIC) and multi-core CPUs. The library is written in C++ and supports CUDA, OpenCL, and OpenMP (including switches at runtime).
- [CUSP](https://github.com/cusplibrary/cusplibrary): A C++ Templated Sparse Matrix Library.
- [Boost uBlas](https://www.boost.org/doc/libs/1_59_0/libs/numeric/ublas/doc/): A C++ template class library that provides BLAS level 1, 2, 3 functionality for dense, packed and sparse matrices. The design and implementation unify mathematical notation via operator overloading and efficient code generation via expression templates.
- [CUV](https://github.com/deeplearningais/CUV): A C++ template and Python library which makes it easy to use NVIDIA(tm) CUDA.
- [Armadillo](https://arma.sourceforge.net/): A high quality linear algebra library (matrix maths) for the C++ language, aiming towards a good balance between speed and ease of use.

## Development Software: Debugging and Profiling
- [Memcheck (Valgrind)](https://valgrind.org/docs/manual/mc-manual.html): A memory error detector.
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html): A performance analysis tool for Linux, Windows, Android, and macOS.
- [gprof](https://hpc.llnl.gov/software/development-environment-software/gprof): A performance analysis tool for Unix applications.
- [FPChecker](https://fpchecker.org/): A tool for detecting floating-point accuracy problems.
- [HPCToolkit](http://hpctoolkit.org/): An integrated suite of tools for measurement and analysis of program performance on computers ranging from multicore desktop systems to the nation's largest supercomputers.

## University Courses & Tutorials
- [HLS Tutorial and Deep Learning Accelerator Design Lab1](https://courses.cs.washington.edu/courses/cse599s/18sp/hw/1.html)
- [UCSB: CS 240A: Applied Parallel Computing](https://sites.cs.ucsb.edu/~tyang/class/240a17/refer.html)
- [UC Berkeley: CS267](https://sites.google.com/lbl.gov/cs267-spr2023)
- [UT Austin: EE382 System-on-Chip (SoC) Design](https://users.ece.utexas.edu/~gerstl/ee382m_f18/labs/lab2.htm)
- [UT Austin (Flame): LAFF-On Programming for High Performance](https://www.cs.utexas.edu/users/flame/laff/pfhp/index.html)

## Selected Papers
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://dl.acm.org/doi/10.1145/2764454). FG Van Zee, RA Van De Geijn. 2015.
- [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://ieeexplore.ieee.org/document/6877334). TM Smith, R Van De Geijn, M Smelyanskiy, JR Hammond, FG Van Zee. 2014.
- [Model-driven Level 3 BLAS Performance Optimization on Loongson 3A Processor](https://ieeexplore.ieee.org/document/6413635). Z Xianyi, W Qian, Z Yunquan. 2012.
- [High-performance implementation of the level-3 BLAS](https://dl.acm.org/doi/10.1145/1377603.1377607). K Goto, R Van De Geijn. 2008.
- [Anatomy of high-performance matrix multiplication](https://dl.acm.org/doi/10.1145/1356052.1356053). K Goto, RA Geijn. 2008.

## Lecture Notes
- [ORNL: CUDA C++ Exercise: Basic Linear Algebra Kernels: GEMM Optimization Strategies](https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final)
- **[Stanford: BLAS-level CPU Performance in 100 Lines of C](https://cs.stanford.edu/people/shadjis/blas.html)**
- [Puedue: Optimizing matrix multiplication](https://www.cs.purdue.edu/homes/grr/cs250/lab6-cache/optimizingMatrixMultiplication.pdf)
- [NJIT: Optimize Matrix Multiplication](https://web.njit.edu/~apv6/courses/hw1.html)

## Blogs
- [Optimizing Matrix Multiplication](https://coffeebeforearch.github.io/2020/06/23/mmul.html)
- [GEMM caching](https://zhuanlan.zhihu.com/p/69700540)
- [Matrix Multiplication on CPU](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Optimizing matrix multiplication: cache + OpenMP](https://www.mgaillard.fr/2020/08/29/matrix-multiplication-optimizing.html)
- **[Tuning matrix multiplication (GEMM) for Intel GPUs](https://www.ibiblio.org/e-notes/webgl/gpu/mul/intel.htm)**
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Building a FAST matrix multiplication algorithm](https://v0dro.in/blog/2018/05/01/building-a-fast-matrix-multiplication-algorithm/)
- [The OpenBLAS Project and Matrix Multiplication Optimization](https://www.leiphone.com/category/yanxishe/Puevv3ZWxn0heoEv.html) (Chinese)
- [Step by step optimization of cuda sgemm](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) (Chinese)
- [OpenBLAS gemm from scratch](https://zhuanlan.zhihu.com/p/65436463)(Chinese)
- [The Proper Approach to CUDA for Beginners: How to Optimize GEMM](https://zhuanlan.zhihu.com/p/478846788) (Chinese)
- [ARMv7 4x4kernel Optimization Practice](https://zhuanlan.zhihu.com/p/333799799) (Chinese)

## Other Learning Resources
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/): New cuBLAS 12.0 Features and Matrix Multiplication Performance on NVIDIA Hopper GPUs.
- [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html): A guide to matrix multiplication performance on NVIDIA GPUs.
- [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html): A programming language for writing highly efficient GPU code.

## Tiny Examples
- [how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm)
- [GEMM](https://github.com/iVishalr/GEMM)
- [BLIS.jl](https://github.com/JuliaLinearAlgebra/BLIS.jl): A  low-level Julia wrapper for BLIS typed interface.
- [blis_apple](https://github.com/xrq-phys/blis_apple): A BLIS library for Apple M1.
- [DGEMM on Int8 Tensor Core](https://github.com/enp1s0/ozIMMU): A library intercepts function calls for cuBLAS DGEMM functions and executes ozIMMU instead.

## How to Contribute
If you have suggestions for adding or removing resources, please feel free to [open a pull request](#) or [create an issue](#).

## License
This work is shared under [MIT License](#).

## Acknowledgments
Special thanks to all the contributors and maintainers of the resources listed in this repository.

