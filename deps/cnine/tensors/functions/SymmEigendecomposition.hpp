/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _SymmEigendecomposition
#define _SymmEigendecomposition

//#include "Tensor.hpp"

#ifdef _WITH_EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>
#endif


namespace cnine{

  template<typename TYPE>
  class SymmEigendecomposition{
  public:

    #ifdef _WITH_EIGEN
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver;

    SymmEigendecomposition(const TensorView<TYPE>& _A){
      //solver(_A){
      solver.compute(Eigen::MatrixXf(_A));
      //solver.compute(_A,Eigen::ComputeThinU|Eigen::ComputeThinV);
    }

    TensorView<TYPE> U() const{
      return solver.eigenvectors();
    }

    TensorView<TYPE> lambda() const{
      return solver.eigenvalues();
    }
    #else

    SymmEigendecomposition(const TensorView<TYPE>& _A){
      CNINE_UNIMPL();
    }

    TensorView<TYPE> U() const{
      CNINE_UNIMPL();
      return TensorView<TYPE>();
    }

    TensorView<TYPE> lambda() const{
      CNINE_UNIMPL();
      return TensorView<TYPE>();
    }

    #endif 

  };

}

#endif 
