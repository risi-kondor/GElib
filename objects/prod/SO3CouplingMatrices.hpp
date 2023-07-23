// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3CouplingMatrices
#define _SO3CouplingMatrices

#include "object_bank.hpp"
#include "DeltaFactor.hpp"
#include "Tensor.hpp"
#include "quadruple_map.hpp"

extern cnine::FFactorial cnine::ffactorial;
extern cnine::DeltaFactor cnine::delta_factor;


namespace GElib{


  class SO3CouplingMatrices: 
    public cnine::object_bank<cnine::quadruple_index<int,int,int,int>,cnine::Tensor<double> >{
  public:

    typedef cnine::object_bank<cnine::quadruple_index<int,int,int,int>,cnine::Tensor<double> > base;

    using base::operator();


    SO3CouplingMatrices(): base([](const cnine::quadruple_index<int,int,int,int>& x){
      auto R=new cnine::Tensor<double>({x.i1+x.i2+1-std::abs(x.i1-x.i2),x.i2+x.i3+1-std::abs(x.i2-x.i3)},cnine::fill_zero());
      auto deltasq=[](const int a, const int b, const int c){return cnine::delta_factor.squared(a,b,c);};
      cnine::DeltaFactor& delta=cnine::delta_factor;

      int j1=x.i1;
      int j2=x.i2;
      int j3=x.i3;
      int j=x.i4;

      //cout<<"Coupling("<<j1<<","<<j2<<","<<j3<<"->"<<j<<")"<<endl;

      int a=j1;
      int b=j2;
      int c=j;
      int d=j3;

      int offs1=std::max(std::abs(j1-j2),std::abs(j-j3));
      int max1=std::min(j1+j2,j+j3);
      vector<cnine::frational> D1(max1+1-offs1);
      for(int e=offs1; e<=max1; e++)
	D1[e-offs1]=delta.squared(a,b,e)*delta.squared(c,d,e);

      int offs2=std::max(std::abs(j2-j3),std::abs(j-j1));
      int max2=std::min(j2+j3,j+j1);
      vector<cnine::frational> D2(max2+1-offs2);
      for(int f=offs2; f<=max2; f++)
	D2[f-offs2]=delta.squared(a,c,f)*delta.squared(b,d,f);

      for(int e=offs1; e<=max1; e++){
	for(int f=offs2; f<=max2; f++){

	  int a1=a+b+e;
	  int a2=c+d+e;
	  int a3=a+c+f;
	  int a4=b+d+f;

	  int b1=a+b+c+d;
	  int b2=a+d+e+f;
	  int b3=b+c+e+f;

	  double w=1;
	  cnine::FFactorial& fact=cnine::ffactorial;
	  int lower=max(a1,max(a2,max(a3,a4)));
	  int upper=min(b1,min(b2,b3));
	  for(int z=lower; z<=upper; z++)
	    w+=(1-2*(z+b1)%2)*
	      (fact(z+1)/(fact(z-a1)*fact(z-a2)*fact(z-a3)*fact(z-a4)*fact(b1-z)*fact(b2-z)*fact(b3-z)));

	  R->set(e-offs1,f-offs2,w*sqrt(D1[e-offs1]*D2[f-offs2]*(2*e+1)*(2*f+1)));
	}
      }

      return R;
    }){}


  public: // ---- Access -------------------------------------------------------------------------------------


    cnine::Tensor<double>& operator()(const int l1, const int l2, const int l3, const int l){
      return (*this)(cnine::quadruple_index<int,int,int,int>(l1,l2,l3,l));
    }

  };

}


#endif 
      //auto R=new cnine::Tensor<double>({x.j1+x.j2+1-std::abs(x.j1-x.j2),x.j2+x.j3+1-std::abs(x.j2-x.j3)},cnine::fill_zero());
    //typedef cnine::quadruple_index<int,int,int,int> CouplingSignature;
  /*
namespace GElib{
  class CouplingSignature{
  public:
    int j1,j2,j3,j;
    CouplingSignature(const int _j1, const int _j2, const int _j3, const int _j): 
      j1(_j1), j2(_j2), j3(_j3), j(_j){}
    bool operator==(const CouplingSignature& x) const{
      return (j1==x.j1)&&(j2==x.j2)&&(j3==x.j3)&&(j==x.j);
    }
  };
}
  */


  /*
namespace std{
  template<>
  struct hash<GElib::CouplingSignature>{
  public:
    size_t operator()(const GElib::CouplingSignature& x) const{
      size_t h=hash<int>()(x.j1);
      h=(h<<1)^hash<int>()(x.j2);
      h=(h<<1)^hash<int>()(x.j3);
      h=(h<<1)^hash<int>()(x.j);
      return h;
    }
  };
}
  */
