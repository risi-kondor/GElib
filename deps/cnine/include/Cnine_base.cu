//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This source code file is subject to the terms of the noncommercial 
//  license distributed with cnine in the file LICENSE.TXT. Commercial 
//  use is prohibited. All redistributed versions of this file, whether 
//  in its original or modified form, must retain this copyright notice 
//  and must be accompanied by a verbatim copy of the license. 


#include "Cnine_base.hpp"

// __device__ __constant__ unsigned char cg_cmem[CNINE_CONST_MEM_SIZE];

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
cublasHandle_t cnine_cublas;
//cublasCreate(&Cengine_cublas);
#endif 
