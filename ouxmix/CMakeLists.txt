cmake_minimum_required(VERSION 3.15)
project(test_ffp LANGUAGES Fortran)

if(NOT DEFINED CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER mpif90)
endif()

if(NOT DEFINED CASE_GPU_CC)
  set(CASE_GPU_CC "89")   # RTX 4060: CC 8.9; override with -DCASE_GPU_CC=XX
endif()

# Inline the EFT helpers so two_prod maps to a single FMAF.F32 instruction.
# -fast is required: enables -Mfma so fma() generates hardware FMA (not a,b,-hi sequence).
set(FFP_INLINE "name:fltflt_two_sum,name:fltflt_fast_two_sum,name:fltflt_two_prod_fma,name:fltflt_add_same_sign,name:fltflt_fma,name:fltflt_fma_approx,name:fltflt_abs,name:fltflt_sqrt,name:fltflt_sqrt_fast,name:fltflt_norm3d,name:fltflt_dot2,name:fltflt_dot3,name:fltflt_add3,name:fltflt_add4,name:fltflt_add5,name:fltflt_square,name:fltflt_recip,name:fltflt_shfl_down,name:fltflt_shfl_xor")
set(CMAKE_Fortran_FLAGS
  "-cuda -fast -Mpreprocess -Minline=${FFP_INLINE} -Minfo=accel,inline -gpu=ptxinfo,rdc,lto,cc${CASE_GPU_CC}")

foreach(op add sub mul div bench dot cmp fma math)
  add_executable(test_${op} ../test/test_${op}.f90 fltflt.f90)
  set_target_properties(test_${op} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

find_program(PYTHON3_EXE python3)
if(PYTHON3_EXE)
  add_custom_target(plot
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_bench
    COMMAND ${PYTHON3_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/../test/plot_ffp.py
            ffp_results.csv ffp_results.png
    DEPENDS test_bench
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../test
    COMMENT "Running benchmark and plotting results -> test/ffp_results.png")
endif()
