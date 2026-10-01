cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 80 CACHE STRING "CUDA SM architecture (A100=80)")

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu )
set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(simplest_kernel PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})

add_executable(sgemm_experiment experiment_runner.cu src/runner.cu)
set_target_properties(sgemm_experiment PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm_experiment ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# 2026-10-01 isolated optimization harness.  Unlike the historical runner,
# profile mode launches only the selected custom kernel, so NCU cannot capture
# the cuBLAS correctness reference by accident.
add_executable(new_opt_runner new_opt_runner.cu src/runner.cu)
set_target_properties(new_opt_runner PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_compile_options(new_opt_runner PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
target_link_libraries(new_opt_runner ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
