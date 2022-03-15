#include <cuda.h>
#include <cuda_runtime.h>
#include "GElib_base.hpp"

__device__ __constant__ unsigned char cg_cmem[CNINE_CONST_MEM_SIZE];
#define _SO3CG_CUDA_CONCAT

#include "SO3partA_CGproduct.cu"
#include "SO3partA_DiagCGproduct.cu"

#include "SO3partB_addCGproduct.cu"
#include "SO3partB_addCGproduct_back0.cu"
#include "SO3partB_addCGproduct_back1.cu"

#include "SO3Fpart_addFproduct.cu"
#include "SO3Fpart_addFproduct_back0.cu"
#include "SO3Fpart_addFproduct_back1.cu"

