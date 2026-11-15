# FmaX test suite.  `make test` builds three drivers and runs run_tests.py:
#   ff_test        FmaX/fltflt.f90 with the production flags (-fast: FMA contraction ON)
#   ff_test_nofma  the same source with FMA contraction OFF (-Mnofma); every
#                  result must be bit-identical to ff_test, otherwise some
#                  error-free transformation depends on compiler contraction
#   ff_matx        NVIDIA MatX fltflt.h (vendored in matx/), the bit-exact reference.
#                  Built with --ftz=true because nvfortran -fast flushes FP32
#                  subnormals to zero on the device; without it the two differ
#                  only where an intermediate lo underflows (|result| ~< 1e-23).
# CASE_GPU_CC selects the GPU architecture (no default on purpose: a wrong one
# builds cleanly and fails at launch). Example: make test CASE_GPU_CC=120
NVFORTRAN   := nvfortran
NVCC        := nvcc
PYTHON      := python3
ifndef CASE_GPU_CC
$(error set CASE_GPU_CC, e.g. make test CASE_GPU_CC=120 (RTX PRO 6000 Blackwell), 89 (RTX 4060), 90 (H100/GH200))
endif
FFP_INLINE  := name:fltflt_two_sum,name:fltflt_fast_two_sum,name:fltflt_two_prod_fma
FCBASE      := -cuda -fast -Mpreprocess -Minline=$(FFP_INLINE) -gpu=ptxinfo,rdc,lto,cc$(CASE_GPU_CC) -Wl,-z,noexecstack
LIB         := ../FmaX/fltflt.f90

.PHONY: all test clean

all: ff_test ff_test_nofma ff_matx

ff_test: $(LIB) ff_test.f90
	mkdir -p build/fma && $(NVFORTRAN) $(FCBASE) -module build/fma -o $@ $(LIB) ff_test.f90

ff_test_nofma: $(LIB) ff_test.f90
	mkdir -p build/nofma && $(NVFORTRAN) $(FCBASE) -Mnofma -module build/nofma -o $@ $(LIB) ff_test.f90

ff_matx: matx/ff_matx.cu matx/fltflt.h
	$(NVCC) -O3 -std=c++20 --ftz=true -arch=sm_$(CASE_GPU_CC) -o $@ matx/ff_matx.cu

test: all
	$(PYTHON) run_tests.py

clean:
	rm -rf ff_test ff_test_nofma ff_matx build *.o *.mod work
