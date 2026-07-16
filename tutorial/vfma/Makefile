# Build tut_vfma.f90.fypp: preprocess with fypp, then compile the result.

FYPP      := fypp
NVFORTRAN := nvfortran

CASE_GPU_CC ?= 89
FCFLAGS := -cuda -fast -Mpreprocess -Minfo=accel,inline \
           -gpu=ptxinfo,rdc,lto,cc$(CASE_GPU_CC)

.PHONY: all preprocess compile clean

all: compile

preprocess: tut_vfma.f90

tut_vfma.f90: tut_vfma.f90.fypp ../../FmaX/vfma.f90.fypp ../../FmaX/vfma_helpers.py
	$(FYPP) -M ../../FmaX -m vfma_helpers $< $@

compile: tut_vfma.o

tut_vfma.o: tut_vfma.f90
	$(NVFORTRAN) $(FCFLAGS) -c $<

clean:
	rm -f tut_vfma.f90 *.o *.mod
