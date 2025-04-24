cmake_minimum_required(VERSION 3.18)
project(CudaMpiExample LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_executable(main main.cu)

target_link_libraries(main PRIVATE MPI::MPI_CXX)
target_include_directories(main PRIVATE /usr/lib/x86_64-linux-gnu/openmpi/include)
