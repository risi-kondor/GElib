#include <cuda.h>
#include <cuda_runtime.h>
#include "GElib_base.hpp"

__device__ __constant__ unsigned char cg_cmem[CNINE_CONST_MEM_SIZE];
