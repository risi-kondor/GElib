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


#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "TensorView_functions.hpp"
#include "Einsum2.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  int niter=10;

  for(int iter=0; iter<niter; iter++){

    TensorView<float> x(dims(4,10),4,0);
    TensorView<float> y(dims(4,3),4,0);

    Einsum2 einsum("iU,iU->iU");
    auto z=einsum(x,y);

    TensorView<float> eps=x.ones_like();
    auto zd=einsum(x+eps,y)-z;
    float delta1=zd.sum();
    cout<<delta1<<endl;

    auto ones=z.ones_like();
    auto xg=x.zeros_like();
    einsum.add_einsum_back0(xg,ones,y);
    float delta2=eps.inp(xg);
    cout<<delta2<<endl;

    TensorView<float> epsb=y.ones_like();
    auto zdb=einsum(x,y+epsb)-z;
    float delta1b=zdb.sum();
    cout<<delta1b<<endl;

    auto yg=y.zeros_like();
    einsum.add_einsum_back1(yg,ones,x);
    float delta2b=epsb.inp(yg);
    cout<<delta2b<<endl;

    cout<<endl;
  }

}
