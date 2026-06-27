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
set(FFP_INLINE "name:two_sum,name:fast_two_sum,name:two_prod,name:shfl_down_ff,name:shfl_xor_ff")
set(CMAKE_Fortran_FLAGS
  "-cuda -fast -Mpreprocess -Minline=${FFP_INLINE} -Minfo=accel,inline -gpu=ptxinfo,rdc,lto,cc${CASE_GPU_CC}")

foreach(op add sub mul div bench)
  add_executable(test_${op} ../test/test_${op}.f90 mod_ffp_eft.f90 mod_ffp.f90)
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
