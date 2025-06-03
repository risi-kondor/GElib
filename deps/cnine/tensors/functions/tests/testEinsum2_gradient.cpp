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

    TensorView<float> x(dims(3,3,3,3),4,0);
    TensorView<float> y(dims(3,3,3,3),4,0);
    auto estr=EinsumForm2::random_string(); 
    //estr="ijkl,ijkl->ijkl";
    vector<int> rdims(EinsumForm2(estr).bcast_ids.size(),3);
    cout<<estr<<endl;

    auto z=einsum(estr,x,y,rdims);

    TensorView<float> eps=x.gaussian_like();
    float delta1=(einsum(estr,x+eps,y,rdims)-z).sum();
    cout<<delta1<<endl;

    auto ones=z.ones_like();
    auto xg=x.zeros_like();
    einsum_add_back0(estr,xg,y,ones);
    float delta2=eps.inp(xg);
    cout<<delta2<<endl;

    TensorView<float> epsb=y.ones_like();
    auto zdb=einsum(estr,x,y+epsb,rdims)-z;
    float delta1b=zdb.sum();
    cout<<delta1b<<endl;

    auto yg=y.zeros_like();
    einsum_add_back1(estr,x,yg,ones);
    float delta2b=epsb.inp(yg);
    cout<<delta2b<<endl;

    cout<<endl;
  }

}
