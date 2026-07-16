NVFORTRAN   := nvfortran
CASE_GPU_CC ?= 89
FFP_INLINE  := name:fltflt_two_sum,name:fltflt_fast_two_sum,name:fltflt_two_prod_fma,name:fltflt_abs,name:fltflt_sqrt,name:fltflt_sqrt_fast,name:dot3_r4,name:dot3_r8,name:dot3_ff,name:fltflt_shfl_down,name:fltflt_shfl_xor
FCFLAGS     := -cuda -fast -Mpreprocess -Minline=$(FFP_INLINE) -Minfo=accel -gpu=ptxinfo,rdc,lto,cc$(CASE_GPU_CC) -Wl,-z,noexecstack
LIB         := ../../FmaX/fltflt.f90

.PHONY: all run clean

all: tut_mul

tut_mul: $(LIB) tut_mul.f90
	$(NVFORTRAN) $(FCFLAGS) -o $@ $(LIB) tut_mul.f90

run: tut_mul
	./tut_mul

clean:
	rm -f tut_mul *.o *.mod
