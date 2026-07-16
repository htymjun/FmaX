# FmaX

## 🎯 Our Mission

In GPU computing, developers are often forced to choose between **elegant code** and **maximum speed**. CUDA Fortran is no exception—writing bare-metal, ultra-fast kernels usually requires writing bloated, hard-to-maintain scalar codes with manual register mapping.

**FmaX changes this.** Our mission is to:
1.  **Elevate CUDA Fortran Usability:** Let you write intuitive, familiar Fortran codes (resembling CPU implementations) and automatically transform them into optimal GPU instructions.
2.  **Democratize High-Precision GPU Computing:** Bring robust multiple/mixed-precision capabilities to CUDA Fortran, removing the complexity of low-level GPU arithmetic.

## 💡 The Name: The FmaX Identity

The name **FmaX** represents the fusion of our core technical pillars:

*   **F** (for **Fortran**): Keeping the elegance, readability, and multi-dimensional array power of Fortran alive on GPUs.
*   **fma** (the core of arithmetic): Refers both to our compile-time **FMA (Fused Multiply-Add)** optimizations and our focus on **Fast Mixed-precision & Arithmetic** models.
*   **X** (inspired by NVIDIA's **MatX**): Directly inspired by the C++ MatX library's philosophy of maximizing hardware-level performance while providing modern, high-level abstractions.
*   **max** (as in **maximize**): Built to **maximize** both developer productivity and GPU hardware efficiency.

## 🚀 Key Features

### 1. Multi-Precision Math (MatX `fltflt.h` Port)
FmaX features a complete, pure-Fortran translation of NVIDIA MatX's `fltflt.h` library, bringing native-feeling multi-layered arithmetic to CUDA Fortran.
*   **Double-Layered Precision Models:** Harness `float_float` (representing numbers as the sum of two FP32s) to run multi-precise simulations directly on GPU hardware.
*   **Designed for Future GPU Architectures:** Modern and upcoming GPU architectures (such as NVIDIA Blackwell and beyond) continue to aggressively widen the throughput gap between lower-precision (FP32/Tensor Cores) and native double-precision (FP64) execution units. By using `float_float` arithmetic, FmaX allows you to emulate high-precision calculations while riding the performance wave of next-generation, high-throughput lower-precision silicon.

### 2. Optimized GPU Code Generation (`fypp`-powered)
No more trade-offs between clean array-syntax and speed. FmaX uses `fypp` (Fortran Preprocessor) to statically analyze and expand your expressions.
*   **Zero-Overhead Abstraction:** Input simple array slices; FmaX automatically unfolds them into static, register-mapped scalar operations.
*   **Smart Register Reuse:** Automatically identifies memory-space overlaps (aliasing) and ensures each data point is loaded into the GPU registers exactly once—completely eliminating redundant global memory (DRAM) traffic.

## License

This project is a CUDA Fortran translation of the C++ header `fltflt.h`
distributed with [NVIDIA MatX](https://github.com/NVIDIA/MatX).

| Work | Copyright |
|------|-----------|
| Original C++ `fltflt.h` | Copyright (c) 2026, NVIDIA Corporation |
| CUDA Fortran translation | Copyright (c) 2026, Jun Hatayama |

Both works are released under the **BSD 3-Clause License**.
The full license text is in the header of [`FmaX/fltflt.f90`](FmaX/fltflt.f90).

## References

- A. Thall (2006). "Extended-Precision Floating-Point Numbers for GPU Computation."
- Y. Zhang & J. Aiken (SC'25). "High-Performance Branch-Free Algorithms for Extended-Precision Floating-Point Arithmetic." *(FPAN addition)*
- T. Ogita, S. M. Rump & S. Oishi (2005). "Accurate Sum and Dot Product." *SIAM J. Sci. Comput.* *(compensated dot)*
