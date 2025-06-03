cmake_minimum_required(VERSION 3.25)
project( cnine VERSION 0.2.0 LANGUAGES CXX)
include(CheckLanguage)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}") # Debug message

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

# Initialize list for CPU source files
set(CNINE_CPU_SOURCES "")

# Add specific core files
list(APPEND CNINE_CPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/Cnine_base.cpp")
list(APPEND CNINE_CPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/external/EigenRoutines.cpp")

# Define cnine's primary source directories
set(CNINE_CORE_CPP_DIRS include external algorithms combinatorial containers hpc math matrices tensors utility wrappers tensor_views "tensors/functions")

# Loop through each directory and glob for .cpp files
foreach(SUBDIR ${CNINE_CORE_CPP_DIRS})
    file(GLOB CPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/*.cpp")
    foreach(CPP_FILE ${CPP_FILES})
        # Exclude test files, files from 'cuda' subdirectories, and already added files
        if(NOT CPP_FILE MATCHES "/tests?/" AND
           NOT CPP_FILE MATCHES "/cuda/" AND
           NOT CPP_FILE STREQUAL "include/Cnine_base.cpp" AND
           NOT CPP_FILE STREQUAL "external/EigenRoutines.cpp")
            list(APPEND CNINE_CPU_SOURCES "${CPP_FILE}")
        endif()
    endforeach()
endforeach()

# Fallback if no CPU sources are found
if(NOT CNINE_CPU_SOURCES)
    message(WARNING "No CPU source files found for cnine library. Adding dummy.cpp as fallback.")
    list(APPEND CNINE_CPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cuda/dummy.cpp")
else()
    message(STATUS "cnine CPU sources: ${CNINE_CPU_SOURCES}")
endif()

# Add the library with the collected source files
add_library(cnine STATIC ${CNINE_CPU_SOURCES})
# target_include_directories(cnine PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>) # Replaced by target_link_libraries with Eigen3::Eigen
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
  legacy/tensor # Added for RtensorObj.hpp and similar
  legacy/backendA # Added for RtensorA.hpp and similar
  legacy/scalar   # Added for RscalarA.hpp and similar
)

foreach(directory IN LISTS cnine_include_dirs)
  target_include_directories(cnine PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${directory}> $<INSTALL_INTERFACE:${directory}>)
endforeach()

target_link_libraries(cnine PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(cnine PUBLIC Eigen3::Eigen) # Modern way to use Eigen

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
