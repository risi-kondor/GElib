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

#ifndef _SingularValueDecomposition
#define _SingularValueDecomposition

#include "TensorView.hpp"

#ifdef _WITH_EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>
#endif


namespace cnine{

  template<typename TYPE>
  class SingularValueDecomposition{
  public:

    Eigen::JacobiSVD<Eigen::MatrixXf> svd;

    SingularValueDecomposition(const TensorView<TYPE>& _A){
      svd.compute(_A,Eigen::ComputeThinU|Eigen::ComputeThinV);
    }

    TensorView<TYPE> U() const{
      return svd.matrixU();
    }

    TensorView<TYPE> S() const{
      return svd.singularValues();
    }

    TensorView<TYPE> V() const{
      return svd.matrixV();

    }


  };

}

#endif 
