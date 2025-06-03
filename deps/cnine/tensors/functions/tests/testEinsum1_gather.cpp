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
#include "Einsum1.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  int niter=1;
  int n=5;

  for(int iter=0; iter<niter; iter++){

    TensorView<float> x(dims(n,n),3,0);
    GatherMapB gmap=GatherMapB::random(n,n);
    cout<<gmap<<endl;

    //string estr="Sj->Sj";
    string estr="jS->Sj";
    vector<int> rdims(EinsumForm1(estr).bcast_ids.size(),3);
    cout<<estr<<endl;

    auto z=einsum(estr,x,gmap,rdims);
    cout<<z<<endl;

    /*
    TensorView<float> eps=x.gaussian_like();
    float delta1=(einsum(estr,x+eps,rdims)-z).sum();
    cout<<delta1<<endl;

    auto ones=z.ones_like();
    auto xg=x.zeros_like();
    einsum_add_back(estr,xg,ones);
    float delta2=eps.inp(xg);
    cout<<delta2<<endl;
    */

    cout<<endl;
  }

}
