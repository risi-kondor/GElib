function(find_torch_python_library OUTPUT_VAR)

  # Finding the right torch_python lib is a bit tricky.
  # Pytorch CMake doesn't provide.
  # We are looking for the path of the critical libc10 library of pytorch.
  # And search that path for the torch_python library
  
  find_package(Torch REQUIRED)
  # Find libc10 path and extract directory
  foreach(lib IN LISTS TORCH_LIBRARIES)
    if(lib MATCHES "/libc10")  
      get_filename_component(LIBC10_DIR "${lib}" DIRECTORY)
      message(STATUS "Found libc10.so directory: ${LIBC10_DIR}")
      break()
    endif()
  endforeach()

  if(NOT LIBC10_DIR)
    message(FATAL_ERROR "Could not find libc10.so in ${TORCH_LIBRARIES}")
  endif()

  # Find torch_python in the same directory
  find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    HINTS ${LIBC10_DIR}
    NO_DEFAULT_PATH  # Only search in this specific directory
    REQUIRED
  )
  
  if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Cannot locate torch python library. Searched in: ${TORCH_PYTHON_PATH}")
  else()
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIBRARY}")
    set(${OUTPUT_VAR} ${TORCH_PYTHON_LIBRARY} PARENT_SCOPE)
  endif()
endfunction()
