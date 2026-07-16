NVFORTRAN   := nvfortran
CASE_GPU_CC ?= 89
FFP_INLINE  := name:fltflt_two_sum,name:fltflt_fast_two_sum,name:fltflt_two_prod_fma,name:fltflt_abs,name:fltflt_sqrt,name:fltflt_sqrt_fast,name:dot3_r4,name:dot3_r8,name:dot3_ff,name:fltflt_shfl_down,name:fltflt_shfl_xor
FCFLAGS     := -cuda -fast -Mpreprocess -Minline=$(FFP_INLINE) -Minfo=accel -gpu=ptxinfo,rdc,lto,cc$(CASE_GPU_CC) -Wl,-z,noexecstack
LIB         := ../../FmaX/fltflt.f90

.PHONY: all run clean

all: tut_div

tut_div: $(LIB) tut_div.f90
	$(NVFORTRAN) $(FCFLAGS) -o $@ $(LIB) tut_div.f90

run: tut_div
	./tut_div

clean:
	rm -f tut_div *.o *.mod
