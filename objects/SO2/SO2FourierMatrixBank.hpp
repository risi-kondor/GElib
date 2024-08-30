// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO2FourierMatrixBank
#define _SO2FourierMatrixBank

#include <mutex>

#include "CtensorB.hpp"


namespace GElib{

  class SO2FourierMatrixBank{
  private:

    typedef pair<int,int> Findex;
    typedef cnine::CtensorB Ctensor;

    mutex safety_mx;
    mutex safety_mxC;

    unordered_map<Findex,Ctensor*> matrices;
    unordered_map<Findex,Ctensor*> matricesC;


  public:

    SO2FourierMatrixBank(){}
    
    SO2FourierMatrixBank(const SO3_CGbank& x)=delete;
    SO2FourierMatrixBank& operator=(const SO3_CGbank& x)=delete;
    
    ~SO2FourierMatrixBank(){
      for(auto p:matrices) delete p.second;
      for(auto p:matricesC) delete p.second;
    }


  public:

    const Ctensor& get(const int m, const int n){
      lock_guard<mutex> lock(safety_mx);
      auto it=matrices.find(pair<int,int>(m,n));
      if(it!=matrices.end()) return *it->second;

      Ctensor* F=new Ctensor(cnine::Gdims({m,n}));
      for(int i=0; i<m; i++){
	float a=M_PI*2.0*i/n;
	for(int j=0; j<n; j++)
	  F->set(i,j,std::exp(complex<float>(0,a*j)));
      }

      matrices[pair<int,int>(n,m)]=F;
      return *F;
    }


    #ifdef _WITH_CUDA
    const Ctensor& getC(const int m, const int n, const int dev=1){
      lock_guard<mutex> lock(safety_mxC);
      auto it=matricesC.find(pair<int,int>(m,n));
      if(it!=matricesC.end()) return *it->second;
      
      Ctensor* F=new Ctensor(get(m,n),dev);
      matricesC[pair<int,int>(n,m)]=F;
      return *F;      
    }
    #endif 

  };

}


#endif 
