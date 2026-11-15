// Reference driver: runs one fltflt operation from NVIDIA MatX's fltflt.h
// (vendored here, pinned to MatX commit a98502025035) over an input file and
// writes the result. test/ff_test.f90 does the same with FmaX/fltflt.f90, and
// test/run_tests.py compares the two bit for bit.
//
//   usage: ff_matx <op> <in.bin> <out.bin>
//
// The op codes and the binary layout are shared with ff_test.f90 (see
// run_tests.py: OPS / write_inputs / read_outputs). An op that MatX does not
// provide (FmaX-only extensions) exits with status 3.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda/std/cmath>   // cuda::std::isfinite, used by fltflt.h

#define __MATX_HOST__ __host__
#define __MATX_DEVICE__ __device__
#define __MATX_INLINE__ __forceinline__
#include "fltflt.h"

using matx::fltflt;

struct In {
  const float *ah, *al, *bh, *bl, *ch, *cl, *dh, *dl;
  const double *x8;
  const int *k;
};

__device__ __forceinline__ fltflt mk(float h, float l) { return fltflt{h, l}; }

__global__ void kern(int op, int n, In in, float *rh, float *rl, int *ri, int *unsup) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const fltflt a = mk(in.ah[i], in.al[i]);
  const fltflt b = mk(in.bh[i], in.bl[i]);
  const fltflt c = mk(in.ch[i], in.cl[i]);
  const float a4 = in.ah[i], b4 = in.bh[i], c4 = in.ch[i];
  fltflt r{0.0f, 0.0f};
  int iv = 0;
  using namespace matx;
  switch (op) {
    case 1:  r = fltflt_add(a, b); break;
    case 2:  r = fltflt_add(a, b4); break;
    case 3:  r = fltflt_add(a4, b); break;
    case 4:  r = fltflt_sub(a, b); break;
    case 5:  r = fltflt_sub(a, b4); break;
    case 6:  r = fltflt_sub(a4, b); break;
    case 7:  r = fltflt_mul(a, b); break;
    case 8:  r = fltflt_mul(a, b4); break;
    case 9:  r = fltflt_mul(a4, b); break;
    case 10: r = fltflt_div(a, b); break;
    case 11: r = fltflt_div(a, b4); break;
    case 12: r = fltflt_div(a4, b); break;
    case 13: r = fltflt_fma(a, b, c); break;
    case 14: r = fltflt_fma(a, b, c4); break;
    case 15: r = fltflt_fma(a4, b, c); break;
    case 16: r = fltflt_fma(a, b4, c); break;
    case 17: r = fltflt_fma(a, b4, c4); break;
    case 18: r = fltflt_fma(a4, b, c4); break;
    case 19: r = fltflt_fma(a4, b4, c); break;
    case 20: r = fltflt_fma_approx(a, b, c); break;
    case 21: r = fltflt_fma_approx(a, b, c4); break;
    case 22: r = fltflt_fma_approx(a4, b, c); break;
    case 23: r = fltflt_fma_approx(a, b4, c); break;
    case 24: r = fltflt_fma_approx(a, b4, c4); break;
    case 25: r = fltflt_fma_approx(a4, b, c4); break;
    case 26: r = fltflt_fma_approx(a4, b4, c); break;
    case 27: r = fltflt_add_same_sign(a, b); break;
    case 28: r = fltflt_add_same_sign(a, b4); break;
    case 29: r = fltflt_add_same_sign(a4, b); break;
    case 30: r = fltflt_sqrt(a); break;
    case 31: r = fltflt_sqrt_fast(a); break;
    case 32: r = fltflt_norm3d(a, b, c); break;
    case 33: r = fltflt_abs(a); break;
    case 34: r = fltflt_round_to_nearest(a); break;
    case 35: r = fltflt_round_toward_zero(a); break;
    case 36: r = fltflt_floor(a); break;
    case 37: r = fltflt_fmod(a, b); break;
    case 38: r = fltflt_fmod(a, b4); break;
    case 39: r = fltflt(in.x8[i]); break;
    case 40: r = -a; break;
    case 41: iv = (a == b); break;
    case 42: iv = (a != b); break;
    case 43: iv = (a < b); break;
    case 44: iv = (a > b); break;
    case 45: iv = (a <= b); break;
    case 46: iv = (a >= b); break;
    case 47: iv = (a == b4); break;
    case 48: iv = (a != b4); break;
    case 49: iv = (a < b4); break;
    case 50: iv = (a > b4); break;
    case 51: iv = (a <= b4); break;
    case 52: iv = (a >= b4); break;
    case 53: iv = (a4 == b); break;
    case 54: iv = (a4 != b); break;
    case 55: iv = (a4 < b); break;
    case 56: iv = (a4 > b); break;
    case 57: iv = (a4 <= b); break;
    case 58: iv = (a4 >= b); break;
    default: if (i == 0) *unsup = 1; break;
  }
  rh[i] = r.hi;
  rl[i] = r.lo;
  ri[i] = iv;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2);} } while (0)

template <class T> T* up(const std::vector<T>& h) {
  T* d; CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: %s op in.bin out.bin\n", argv[0]); return 1; }
  const int op = atoi(argv[1]);
  FILE* f = fopen(argv[2], "rb");
  if (!f) { perror(argv[2]); return 1; }
  int32_t n;
  if (fread(&n, 4, 1, f) != 1) return 1;
  std::vector<float> v[8];
  for (int j = 0; j < 8; ++j) { v[j].resize(n); if (fread(v[j].data(), 4, n, f) != (size_t)n) return 1; }
  std::vector<double> x8(n);
  if (fread(x8.data(), 8, n, f) != (size_t)n) return 1;
  std::vector<int> k(n);
  if (fread(k.data(), 4, n, f) != (size_t)n) return 1;
  fclose(f);

  In in{up(v[0]), up(v[1]), up(v[2]), up(v[3]), up(v[4]), up(v[5]), up(v[6]), up(v[7]),
        up(x8), up(k)};
  float *rh, *rl; int *ri, *unsup;
  CK(cudaMalloc(&rh, n * 4)); CK(cudaMalloc(&rl, n * 4)); CK(cudaMalloc(&ri, n * 4));
  CK(cudaMalloc(&unsup, 4)); CK(cudaMemset(unsup, 0, 4));
  kern<<<(n + 127) / 128, 128>>>(op, n, in, rh, rl, ri, unsup);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  int u; CK(cudaMemcpy(&u, unsup, 4, cudaMemcpyDeviceToHost));
  if (u) return 3;
  std::vector<float> hh(n), hl(n); std::vector<int> hi(n);
  CK(cudaMemcpy(hh.data(), rh, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hl.data(), rl, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hi.data(), ri, n * 4, cudaMemcpyDeviceToHost));
  FILE* g = fopen(argv[3], "wb");
  fwrite(hh.data(), 4, n, g); fwrite(hl.data(), 4, n, g); fwrite(hi.data(), 4, n, g);
  fclose(g);
  return 0;
}
