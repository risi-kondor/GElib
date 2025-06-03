cmake_minimum_required(VERSION 3.25)
project( cnine VERSION 0.2.0 LANGUAGES CXX)
include(CheckLanguage)

option(WITH_CUDA "Compiling with CUDA support" OFF)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WITH_CUDA)
  check_language(CUDA)
  # Set the C++ standard
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 11...<13 COMPONENTS CUDA::cudart CUDA::cublas)

  else(CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA compiler found")
    set(WITH_CUDA OFF)
  endif(CMAKE_CUDA_COMPILER)
  if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit, continue with CUDA built.")
  else(CUDAToolkit_FOUND)
    message(STATUS "No CUDA Toolkit found, built without CUDA. Please ensure the installed torch version is built without CUDA as well.")
    set(WITH_CUDA OFF)
  endif(CUDAToolkit_FOUND)
endif(WITH_CUDA)
  
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(cnine STATIC cuda/dummy.cpp)
set_target_properties(cnine PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(cnine_include_dirs
  include
  algorithms
  combinatorial
  containers
  hpc
  math
  matrices
  tensors
  utility
  wrappers
  tensor_views
  tensors/functions
)

foreach(directory IN LISTS cnine_include_dirs)
  target_include_directories(cnine PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${directory}> $<INSTALL_INTERFACE:${directory}>)
endforeach()

target_link_libraries(cnine PUBLIC "${TORCH_LIBRARIES}")

target_compile_definitions(cnine PUBLIC _WITH_ATEN)
target_compile_definitions(cnine PUBLIC CNINE_RANGE_CHECKING)
target_compile_definitions(cnine PUBLIC CNINE_SIZE_CHECKING)
target_compile_definitions(cnine PUBLIC CNINE_DEVICE_CHECKING)
target_compile_definitions(cnine PUBLIC WITH_FAKE_GRAD)

if(WITH_CUDA)
  target_compile_definitions(cnine PUBLIC _WITH_CUDA)
  target_compile_definitions(cnine PUBLIC _WITH_CUBLAS)
  target_compile_definitions(cnine PUBLIC WITH_FAKE_GRAD)
endif(WITH_CUDA)

target_compile_options(cnine PUBLIC -Wno-sign-compare)
target_compile_options(cnine PUBLIC -Wno-deprecated-declarations)
target_compile_options(cnine PUBLIC -Wno-unused-variable)
target_compile_options(cnine PUBLIC -Wno-unused-but-set-variable)
target_compile_options(cnine PUBLIC -Wno-reorder)
target_compile_options(cnine PUBLIC -Wno-reorder-ctor)

if(WITH_CUDA)
  add_subdirectory(cuda)
endif(WITH_CUDA)

option(BUILD_PYTHON "Build the python bindings" OFF)
if(BUILD_PYTHON)
  add_subdirectory(python)
endif(BUILD_PYTHON)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cnineConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/cnineConfig.cmake  INSTALL_DESTINATION lib/cmake/cnine)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/cnineVersion.cmake  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

# Install the library
install(TARGETS cnine  EXPORT cnine  LIBRARY DESTINATION lib  ARCHIVE DESTINATION lib INCLUDES DESTINATION cnine/include)

# Install the headers
foreach(directory IN LISTS cnine_include_dirs)
  install(DIRECTORY ${directory}/ DESTINATION ${directory})
endforeach()

# Install the config files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cnineConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/cnineVersion.cmake  DESTINATION lib/cmake/cnine)

# Install the export
install(EXPORT cnine FILE cnineTargets.cmake NAMESPACE cnine:: DESTINATION lib/cmake/cnine)

# Register package in user's package registry
export(PACKAGE cnine)
