cmake_minimum_required(VERSION 3.15)
project(tutorial_ffp LANGUAGES Fortran)

if(NOT DEFINED CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER mpif90)
endif()

if(NOT DEFINED CASE_GPU_CC)
  set(CASE_GPU_CC "89")   # RTX 4060: CC 8.9; override with -DCASE_GPU_CC=XX
endif()

# Inline EFT helpers for optimal GPU performance (same flags as ouxmix/).
set(FFP_INLINE "name:two_sum,name:fast_two_sum,name:two_prod,name:shfl_down_ff,name:shfl_xor_ff")
set(CMAKE_Fortran_FLAGS
  "-cuda -fast -Mpreprocess -Minline=${FFP_INLINE} -Minfo=accel,inline -gpu=ptxinfo,rdc,lto,cc${CASE_GPU_CC}")

set(LIB ../ouxmix/mod_ffp_eft.f90 ../ouxmix/mod_ffp.f90)

foreach(op add sub mul div add_r8 sub_r8 mul_r8 div_r8)
  add_executable(tut_${op} tut_${op}.f90 ${LIB})
  set_target_properties(tut_${op} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
