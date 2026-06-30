cmake_minimum_required(VERSION 3.15)
project(tutorial_ffp LANGUAGES Fortran)

if(NOT DEFINED CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER mpif90)
endif()

if(NOT DEFINED CASE_GPU_CC)
  set(CASE_GPU_CC "89")   # RTX 4060: CC 8.9; override with -DCASE_GPU_CC=XX
endif()

# Inline EFT helpers for optimal GPU performance (same flags as ouxmix/).
set(FFP_INLINE "name:fltflt_two_sum,name:fltflt_fast_two_sum,name:fltflt_two_prod_fma,name:fltflt_abs,name:fltflt_sqrt,name:fltflt_sqrt_fast,name:dot3_r4,name:dot3_r8,name:dot3_ff,name:fltflt_shfl_down,name:fltflt_shfl_xor")
set(CMAKE_Fortran_FLAGS
  "-cuda -fast -Mpreprocess -Minline=${FFP_INLINE} -Minfo=accel,inline -gpu=ptxinfo,rdc,lto,cc${CASE_GPU_CC}")

set(LIB ../ouxmix/fltflt.f90)

foreach(op add sub mul div add_r8 sub_r8 mul_r8 div_r8 dot3)
  add_executable(tut_${op} tut_${op}.f90 ${LIB})
  set_target_properties(tut_${op} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
