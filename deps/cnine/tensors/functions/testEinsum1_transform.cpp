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
#include "TensorTransform1.hpp"

using namespace cnine;

namespace cnine{

  template<typename TYPE>
  class ExampleTransform1: public TensorTransform1<TYPE>{
  public:

    void operator()(TYPE* xarr, TYPE* rarr, const int xn0, const int xs0, const int rn0, const int rs0){
      for(int i=0; i<xn0; i++){
	*(rarr+i*rs0)+=pow(*(xarr+i*xs0),2);
      }
    }

  };

}


int main(int argc, char** argv){

  cnine_session session;
  int niter=1;
  int n=5;

  for(int iter=0; iter<niter; iter++){

    TensorView<float> x(dims(n,n),3,0);
    ExampleTransform transf;
    GatherMapB gmap=GatherMapB::random(n,n);
    cout<<gmap<<endl;

    //string estr="Sj->Sj";
    string estr="Fj->Fj";
    vector<int> rdims(EinsumForm1(estr).bcast_ids.size(),3);
    cout<<estr<<endl;

    auto z=einsum_transform(estr,x,transf,rdims);
    cout<<z<<endl;


  }

}

