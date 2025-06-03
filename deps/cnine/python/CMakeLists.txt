find_package(Python COMPONENTS Interpreter Development Development.Module REQUIRED)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cnine_base bindings/cnine_py.cpp)
include(../cmake/FindTorchPython.cmake)
find_torch_python_library(TORCH_PYTHON_LIBRARY)
target_link_libraries(cnine_base PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} pybind11::pybind11)
target_link_libraries(cnine_base PUBLIC cnine)
if(WITH_CUDA)
  target_link_libraries(cnine_base PRIVATE CUDA::cudart CUDA::cublas)
endif(WITH_CUDA)

install(TARGETS cnine_base DESTINATION . COMPONENT cnine::python)
