# Awesome GEMM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![awesome-gemm-banner](./img/awesome-gemm-banner.PNG)

> **🚀 Welcome to Awesome GEMM!**  
> A curated and continually evolving list of frameworks, libraries, tutorials, and tools for optimizing **General Matrix Multiply (GEMM)** operations. Whether you're a beginner eager to learn the fundamentals, a developer optimizing performance-critical code, or a researcher pushing the limits of hardware, this repository is your launchpad to mastery.

---

## Why GEMM Matters 💡

General Matrix Multiply is at the core of a wide range of computational tasks: from scientific simulations and signal processing to modern AI workloads like neural network training and inference. Efficiently implementing and optimizing GEMM can lead to dramatic performance improvements across entire systems.

**This repository is a comprehensive resource for:**
- **Students & Beginners:** Learn the basics and theory of matrix multiplication.
- **Engineers & Developers:** Discover frameworks, libraries, and tools to optimize GEMM on CPUs, GPUs, and specialized hardware.
- **Researchers & Performance Experts:** Explore cutting-edge techniques, research papers, and advanced optimization strategies.

---

## Quickstart & Highlights 🌱

If you’re new and just want to dive in, start here:

- **For Beginners:**  
  - [NumPy](https://github.com/numpy/numpy) (CPU, Python) - The go-to library for basic matrix operations.
  - [How To Optimize GEMM](https://github.com/flame/how-to-optimize-gemm) - A step-by-step guide to improving performance from a naive implementation.

- **For GPU Developers:**  
  - [NVIDIA cuBLAS](https://developer.nvidia.com/cublas) - Highly optimized BLAS for NVIDIA GPUs.
  - [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - Templates and building blocks to write your own CUDA GEMM kernels.

- **For Low-Precision & AI Workloads:**  
  - [FBGEMM](https://github.com/pytorch/FBGEMM) (Meta) - Specialized low-precision GEMM for server inference.
  - [gemmlowp](https://github.com/google/gemmlowp) (Google) - Low-precision (integer) GEMM for efficient ML inference.

---

## Table of Contents 📑

- [Fundamental Theories and Concepts 🧠](#fundamental-theories-and-concepts-)
- [General Optimization Techniques 🚀](#general-optimization-techniques-)
- [Frameworks and Development Tools 🛠️](#frameworks-and-development-tools-)
- [Libraries 🗂️](#libraries-)
  - [CPU Libraries 💻](#cpu-libraries-)
  - [GPU Libraries ⚡](#gpu-libraries-)
  - [Cross-Platform Libraries 🌍](#cross-platform-libraries-)
  - [Language-Specific Libraries 🔤](#language-specific-libraries-)
- [Development Software: Debugging and Profiling 🔍](#development-software-debugging-and-profiling-)
- [Learning Resources 📚](#learning-resources-)
  - [University Courses & Tutorials 🎓](#university-courses--tutorials-)
  - [Selected Papers 📝](#selected-papers-)
  - [Lecture Notes 📖](#lecture-notes)
  - [Blogs 🖋️](#blogs-)
  - [Other Resources 🔗](#other-resources)
- [Example Implementations 💡](#example-implementations-)
- [Contributions 🤝](#contributions-)
- [License 📜](#license-)

---

## Fundamental Theories and Concepts 🧠

- **What is GEMM?**  
  - [General Matrix Multiply (Intel)](https://www.intel.com/content/dam/develop/external/us/en/documents/intel-ocl-gemm.pdf) - Intro from Intel.
  - [Spatial-lang GEMM](https://spatial-lang.org/gemm) - High-level overview.

- **Matrix Multiplication Algorithms:**  
  - [Strassen's Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm) - Faster asymptotic complexity for large matrices.  
  - [Winograd's Algorithm](https://en.wikipedia.org/wiki/Winograd_algorithm) - Reduced multiplication count for improved performance.

---

## General Optimization Techniques 🚀

- [How To Optimize GEMM](https://github.com/flame/how-to-optimize-gemm) - Hands-on optimization guide.
- [GEMM: From Pure C to SSE Optimized Micro Kernels](https://www.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html) - Detailed tutorial on going from naive to vectorized implementations.

---

## Frameworks and Development Tools 🛠️

- [BLIS](https://github.com/flame/blis) - A modular framework for building high-performance BLAS-like libraries.
- [BLISlab](https://github.com/flame/blislab) - Educational framework for experimenting with BLIS-like GEMM algorithms.
- [Tensile](https://github.com/ROCm/Tensile) - AMD ROCm JIT compiler for GPU kernels, specializing in GEMM and tensor contractions.

---

## Libraries 🗂️

### CPU Libraries 💻

- [BLASFEO: Optimized for small- to medium-sized dense matrices](https://github.com/giaf/blasfeo) (BSD-2-Clause)
- [blis_apple: BLIS optimized for Apple M1](https://github.com/xrq-phys/blis_apple) (BSD-3-Clause)
- [FBGEMM: Meta's CPU GEMM for optimized server inference](https://github.com/pytorch/FBGEMM) (BSD-3-Clause)
- [gemmlowp: Google's low-precision GEMM library](https://github.com/google/gemmlowp) (Apache-2.0)
- [Intel MKL: Highly optimized math routines for Intel CPUs](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) (Intel Proprietary)
- [libFLAME: High-performance dense linear algebra library](https://github.com/flame/libflame) (BSD-3-Clause)
- [LIBXSMM: Specializing in small/micro GEMM kernels](https://github.com/hfp/libxsmm) (BSD-3-Clause)
- [OpenBLAS: Optimized BLAS implementation based on GotoBLAS2](https://github.com/xianyi/OpenBLAS) (BSD-3-Clause)

### GPU Libraries ⚡

- [clBLAS: BLAS functions on OpenCL for portability](https://github.com/clMathLibraries/clBLAS) (Apache-2.0)
- [CLBlast: Tuned OpenCL BLAS library](https://github.com/CNugteren/CLBlast) (Apache-2.0)
- [hipBLAS: BLAS for AMD GPU platforms (ROCm)](https://github.com/ROCm/hipBLAS) (MIT)
- [hipBLASLt: Lightweight BLAS library on ROCm](https://github.com/ROCm/hipBLASLt) (MIT)
- [NVIDIA cuBLAS: Highly tuned BLAS for NVIDIA GPUs](https://developer.nvidia.com/cublas) (NVIDIA License)
- [NVIDIA cuDNN: Deep learning primitives, including GEMM](https://developer.nvidia.com/cudnn) (NVIDIA License)
- [NVIDIA cuSPARSE: Sparse matrix computations on NVIDIA GPUs](https://developer.nvidia.com/cusparse) (NVIDIA License)
- [NVIDIA CUTLASS: Template library for CUDA GEMM kernels](https://github.com/NVIDIA/cutlass) (BSD-3-Clause)

### Cross-Platform Libraries 🌍

- [ARM Compute Library: Optimized for ARM platforms](https://github.com/ARM-software/ComputeLibrary) (Apache-2.0/MIT)
- [CUSP: C++ templates for sparse linear algebra](https://github.com/cusplibrary/cusplibrary) (Apache-2.0)
- [CUV: C++/Python for CUDA-based vector/matrix ops](https://github.com/deeplearningais/CUV)
- [Ginkgo: High-performance linear algebra on many-core systems](https://github.com/ginkgo-project/ginkgo) (BSD-3-Clause)
- [LAPACK: Foundational linear algebra routines](https://www.netlib.org/lapack/) (BSD-3-Clause)
- [MAGMA: High-performance linear algebra on GPUs and multicore CPUs](https://github.com/icl-utk-edu/magma) (BSD-3-Clause)
- [oneDNN (MKL-DNN): Cross-platform deep learning primitives with optimized GEMM](https://github.com/oneapi-src/oneDNN) (Apache-2.0)
- [viennacl-dev: OpenCL-based linear algebra library](https://github.com/viennacl/viennacl-dev) (MIT)

### Language-Specific Libraries 🔤

**Python:**
- [JAX](https://github.com/google/jax) (Apache-2.0)
- [NumPy](https://github.com/numpy/numpy) (BSD-3-Clause)
- [PyTorch](https://github.com/pytorch/pytorch) (BSD-3-Clause)
- [SciPy](https://github.com/scipy/scipy) (BSD-3-Clause)
- [TensorFlow](https://github.com/tensorflow/tensorflow) (Apache-2.0) & [XLA](https://www.tensorflow.org/xla)

**C++:**
- [Armadillo](https://arma.sourceforge.net/) (Apache-2.0/MIT)
- [Blaze](https://bitbucket.org/blaze-lib/blaze/) (BSD-3-Clause)
- [Boost uBlas](https://www.boost.org/doc/libs/release/libs/numeric/ublas/) (Boost License)
- [Eigen](https://gitlab.com/libeigen/eigen) (MPL2)

**Julia:**
- [BLIS.jl](https://github.com/JuliaLinearAlgebra/BLIS.jl) (BSD-3-Clause)
- [GemmKernels.jl](https://github.com/JuliaGPU/GemmKernels.jl) (BSD-3-Clause)

---

## Development Software: Debugging and Profiling 🔍

**Intel Tools:**
- [Intel Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/advisor.html)
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

**NVIDIA Tools:**
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Nsight Visual Studio Edition](https://developer.nvidia.com/nsight-visual-studio-edition)
- [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/)

**ROCm Tools:**
- [ROCm Profiler (rocprofiler)](https://github.com/ROCm/rocprofiler)

**Others:**
- [Extrae](https://tools.bsc.es/extrae)
- [FPChecker](https://github.com/LLNL/FPChecker)
- [gprof](https://sourceware.org/binutils/docs/gprof/)
- [gprofng](https://sourceware.org/binutils/docs/gprofng.html)
- [HPCToolkit](https://gitlab.com/hpctoolkit/hpctoolkit)
- [LIKWID](https://github.com/RRZE-HPC/likwid)
- [MegPeak](https://github.com/MegEngine/MegPeak)
- [Perf (Linux)](https://perf.wiki.kernel.org/)
- [TAU](https://www.cs.uoregon.edu/research/tau/home.php)
- [VAMPIR](https://vampir.eu/)
- [Valgrind (Memcheck)](https://valgrind.org/docs/manual/mc-manual.html)

---

## Learning Resources 📚

### University Courses & Tutorials 🎓

- [GPU MODE YouTube Channel](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA)
- [HLS Tutorial & Deep Learning Accelerator Lab1](https://courses.cs.washington.edu/courses/cse599s/18sp/hw/1.html)
- [HPC Garage](https://github.com/hpcgarage)
- [MIT OCW: 6.172 Performance Engineering](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/)
- [MIT: Optimizing Matrix Multiplication (6.172 Lecture Notes)](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/6-172-fall-2018/lecture-notes/)
- [NJIT: Optimize Matrix Multiplication](https://web.njit.edu/~apv6/courses/hw1.html)
- [Optimizing Matrix Multiplication using SIMD and Parallelization](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/6-172-fall-2018/lecture-notes/MIT6_172F18_lec5.pdf)
- [ORNL: CUDA C++ Exercise: Basic Linear Algebra Kernels: GEMM Optimization Strategies](https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final)
- [Purdue: Optimizing Matrix Multiplication](https://www.cs.purdue.edu/homes/grr/cs250/lab6-cache/optimizingMatrixMultiplication.pdf)
- [Stanford: BLAS-level CPU Performance in 100 Lines of C](https://cs.stanford.edu/people/shadjis/blas.html)
- [UC Berkeley: CS267 Parallel Computing](https://sites.google.com/lbl.gov/cs267-spr2023)
- [UCSB CS 240A: Applied Parallel Computing](https://sites.cs.ucsb.edu/~tyang/class/240a17/refer.html)
- [UT Austin: LAFF-On Programming for High Performance](https://www.cs.utexas.edu/users/flame/laff/pfhp/index.html)

### Selected Papers 📝

- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality (2015)](https://dl.acm.org/doi/10.1145/2764454)
- [Anatomy of High-Performance Many-Threaded Matrix Multiplication (2014)](https://ieeexplore.ieee.org/document/6877334)
- [Model-driven BLAS Performance on Loongson (2012)](https://ieeexplore.ieee.org/document/6413635)
- [High-performance Implementation of the Level-3 BLAS (2008)](https://dl.acm.org/doi/10.1145/1377603.1377607)
- [Anatomy of High-Performance Matrix Multiplication (2008)](https://dl.acm.org/doi/10.1145/1356052.1356053)

### Blogs 🖋️

- [Building a FAST Matrix Multiplication Algorithm](https://v0dro.in/blog/2018/05/01/building-a-fast-matrix-multiplication-algorithm/)
- [CUDA GEMM Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)
- [CUDA Learn Notes](https://github.com/DefTruth/CUDA-Learn-Notes)
- [CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
- [Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/)
- [Distributed GEMM - A novel CUTLASS-based implementation of Tensor Parallelism for NVLink-enabled systems](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b)
- [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Matrix Multiplication Background Guide (NVIDIA)](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [Matrix Multiplication on CPU](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Matrix-Matrix Product Experiments with BLAZE](https://www.mathematik.uni-ulm.de/~lehn/test_blaze/index.html)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: A Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Optimizing Matrix Multiplication](https://coffeebeforearch.github.io/2020/06/23/mmul.html)
- [Optimizing Matrix Multiplication: Cache + OpenMP](https://www.mgaillard.fr/2020/08/29/matrix-multiplication-optimizing.html)
- [perf-book by Denis Bakhvalov](https://github.com/dendibakh/perf-book)
- [Tuning Matrix Multiplication (GEMM) for Intel GPUs](https://www.ibiblio.org/e-notes/webgl/gpu/mul/intel.htm)
- [Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

---

## Example Implementations 💡

- [chgemm: Int8 GEMM implementations](https://github.com/tpoisonooo/chgemm)
- [CoralGemm: AMD high-performance GEMM implementations](https://github.com/AMD-HPC/CoralGemm) (MIT)
- [CUTLASS-based Grouped GEMM: Efficient grouped GEMM operations](https://github.com/tgale96/grouped_gemm) (Apache-2.0)
- [DeepBench](https://github.com/baidu-research/DeepBench) (Apache-2.0)
- [how-to-optimize-gemm (row-major matmul)](https://github.com/tpoisonooo/how-to-optimize-gemm) (GPLv3)
- [SGEMM_CUDA: Step-by-Step Optimization](https://github.com/siboehm/SGEMM_CUDA) (MIT)
- [simple-gemm](https://github.com/williamfgc/simple-gemm) (MIT)
- [TK-GEMM: a Triton FP8 GEMM kernel using SplitK parallelization](https://pytorch.org/blog/accelerating-llama3/)
- [Toy HGEMM (Tensor Cores with MMA/WMMA)](https://github.com/DefTruth/hgemm-tensorcores-mma) (GPLv3)

---

## Contributions 🤝

We welcome and encourage contributions! You can help by:

- Adding new libraries, tools, or tutorials.
- Submitting performance benchmarks or example implementations.
- Improving documentation or correcting errors.

Submit a pull request or open an issue to get started!

---

## License 📜

This repository is licensed under the [MIT License](LICENSE).

---

*By maintaining this curated list, we hope to empower the community to learn, implement, and optimize GEMM efficiently. Thanks for visiting, and happy computing!*
