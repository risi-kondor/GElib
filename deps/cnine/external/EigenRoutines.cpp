/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineEigenRoutines
#define _CnineEigenRoutines

#include <Eigen/Dense>

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor2_view.hpp" // Parameters are views
#include "Rtensor1_view.hpp" // Parameters are views
#include "TensorView.hpp"    // Return types are TensorView<float>


namespace cnine{


  pair<TensorView<float>,TensorView<float>> eigen_eigendecomp(const Rtensor2_view& x){
    int d=x.n0;
    assert(d==x.n1);
    Eigen::MatrixXd M(d,d);
    for(int i=0; i<d; i++) for(int j=0; j<d; j++) M(i,j)=x(i,j);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    auto Ue=solver.eigenvectors();
    TensorView<float> U(Gdims({d,d}), 1); // fcode=1 for raw/uninitialized
    for(int i=0; i<d; i++) for(int j=0; j<d; j++) U(i,j) = Ue(i,j); // Use operator() for assignment
    auto De=solver.eigenvalues();
    TensorView<float> D(Gdims({x.n0}), 1); // fcode=1 for raw/uninitialized
    for(int i=0; i<d; i++) D(i) = De(i); // Use operator() for assignment
    return make_pair(U,D);
  }


  TensorView<float> eigen_linsolve(const Rtensor2_view& _A, const Rtensor1_view& _b){
    int I=_A.n0;
    int J=_A.n1;
    assert(I==_b.n0);

    Eigen::MatrixXd A(I,J);
    for(int i=0; i<I; i++) for(int j=0; j<J; j++) A(i,j)=_A(i,j);
    Eigen::VectorXd b(I);
    for(int i=0; i<I; i++) b(i)=_b(i);

    TensorView<float> _x(Gdims({J}), 1); // fcode=1 for raw/uninitialized
    auto x=A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    for(int i=0; i<J; i++) _x(i) = x(i); // Use operator() for assignment

    return _x;
  }


}

#endif 
