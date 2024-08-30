// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3FourierMatrixBank
#define _SO3FourierMatrixBank

#include <mutex>

#include "CtensorB.hpp"
#include "WignerMatrix.hpp"


namespace GElib{

  class SO3FourierMatrixBank{
  private:

    typedef pair<int,int> Findex;
    typedef cnine::CtensorB Ctensor;

    mutex safety_mx;
    mutex safety_mxC;

    unordered_map<Findex,Ctensor*> Fmatrices;
    unordered_map<Findex,Ctensor*> FmatricesC;
    unordered_map<Findex,Ctensor*> iFmatrices;
    unordered_map<Findex,Ctensor*> iFmatricesC;
    unordered_map<Findex,Ctensor*> Dmatrices;
    unordered_map<Findex,Ctensor*> DmatricesC;


  public:

    SO3FourierMatrixBank(){}
    
    SO3FourierMatrixBank(const SO3_CGbank& x)=delete;
    SO3FourierMatrixBank& operator=(const SO3_CGbank& x)=delete;
    
    ~SO3FourierMatrixBank(){
      for(auto p:Fmatrices) delete p.second;
      //for(auto p:FmatricesC) delete p.second;
      for(auto p:iFmatrices) delete p.second;
      //for(auto p:iFmatricesC) delete p.second;
      for(auto p:Dmatrices) delete p.second;
      //for(auto p:DmatricesC) delete p.second; // trouble!
    }


  public:

    const Ctensor& Fmatrix(const int l, const int n, const int dev=0){

      if(dev==1){
	lock_guard<mutex> lock(safety_mxC);
	auto it=FmatricesC.find(pair<int,int>(l,n));
	if(it!=FmatricesC.end()) return *it->second;
      
	Ctensor* F=new Ctensor(Fmatrix(l,n,0),dev);
	FmatricesC[pair<int,int>(l,n)]=F;
	return *F;      
      }

      lock_guard<mutex> lock(safety_mx);
      auto it=Fmatrices.find(pair<int,int>(l,n));
      if(it!=Fmatrices.end()) return *it->second;

      Ctensor* F=new Ctensor(cnine::Gdims({2*l+1,n}));
      float fact=1.0/sqrt(n);
      for(int i=0; i<n; i++){
	float a=M_PI*2.0*i/n;
	for(int m=-l; m<=l; m++)
	  F->set(m+l,i,fact*std::exp(complex<float>(0,m*a)));
      }

      Fmatrices[pair<int,int>(l,n)]=F;
      return *F;
    }

    
    const Ctensor& iFmatrix(const int l, const int n, const int dev=0){

      if(dev==1){
	lock_guard<mutex> lock(safety_mxC);
	auto it=iFmatricesC.find(pair<int,int>(l,n));
	if(it!=iFmatricesC.end()) return *it->second;
      
	Ctensor* F=new Ctensor(iFmatrix(l,n,0),dev);
	iFmatricesC[pair<int,int>(l,n)]=F;
	return *F;      
      }

      lock_guard<mutex> lock(safety_mx);
      auto it=iFmatrices.find(pair<int,int>(l,n));
      if(it!=iFmatrices.end()) return *it->second;

      Ctensor* F=new Ctensor(cnine::Gdims({n,2*l+1}));
      float fact=1.0/sqrt(n);
      for(int i=0; i<n; i++){
	float a=M_PI*2.0*i/n;
	for(int m=-l; m<=l; m++)
	  F->set(i,m+l,fact*std::exp(complex<float>(0,-m*a)));
      }

      iFmatrices[pair<int,int>(l,n)]=F;
      return *F;
    }

    
    const Ctensor& Dmatrix(const int l, const int n, const int dev=0){
      pair<int,int> ix(l,n);

      if(dev==1){
	lock_guard<mutex> lock(safety_mxC);
	auto it=DmatricesC.find(ix);
	if(it!=DmatricesC.end()) return *it->second;
      
	Ctensor* D=new Ctensor(Dmatrix(l,n,0),dev);
	DmatricesC[ix]=D;
	return *D;      
      }

      lock_guard<mutex> lock(safety_mx);
      auto it=Dmatrices.find(ix);
      if(it!=Dmatrices.end()) return *it->second;

      Ctensor* D=new Ctensor(cnine::Gdims({2*l+1,n,2*l+1}));
      WignerMatrix<float> Wigner;
      float fact=sqrt(2*l+1)/sqrt(n); ///sqrt(M_PI*8);
      for(int i=0; i<n; i++){
	float theta=M_PI*i/n;
	float fact2=fact*sqrt(sin(theta));
	for(int m1=-l; m1<=l; m1++)
	  for(int m2=-l; m2<=l; m2++)
	    D->set(m1+l,i,m2+l,fact2*Wigner.littled(l,m2,m1,theta));
      }
      Dmatrices[ix]=D;
      return *D;
    }

  };

}


#endif 
