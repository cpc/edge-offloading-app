cmake_minimum_required(VERSION 3.22)
project(bandwidth_tests C)

set(CMAKE_C_STANDARD 99)

set(GIT_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
set(NATIVE_DIR ${GIT_ROOT}/android/app/src/main/cpp)

message(WARNING "native dir: ${NATIVE_DIR}")

add_executable(bandwidth_tests main.c  ${NATIVE_DIR}/sharedUtils.h ${NATIVE_DIR}/platform.h bandwidth_tests.h bandwidth_tests.c )
target_include_directories(bandwidth_tests PUBLIC ${NATIVE_DIR})
#target_include_directories(bandwidth_tests PUBLIC ${NATIVE_DIR} /home/rabijl/Projects/pocl/include/CL/ )

add_library(libpocl SHARED IMPORTED)

#if(NOT DEFINED POCL_DIR)
#    set_target_properties(libpocl PROPERTIES IMPORTED_LOCATION /home/rabijl/Projects/pocl/cmake-build-remote-client/lib/CL/libOpenCL.so)
#endif(NOT DEFINED POCL_DIR)
#target_link_libraries(bandwidth_tests libpocl)

find_package(OpenCL REQUIRED)
target_link_libraries(bandwidth_tests OpenCL)