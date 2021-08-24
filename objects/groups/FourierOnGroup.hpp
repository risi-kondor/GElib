
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _FourierOnGroup
#define _FourierOnGroup

#include "CtensorObj_funs.hpp"

#include "Group.hpp"

namespace GElib{

  template<typename GROUP, typename TENSOR>
  class FourierOnGroup{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::CtensorObj ctensor;

    typedef decltype(GROUP::dummy_element()) ELEMENT; 
    typedef decltype(GROUP::dummy_irrep()) IRREP; 
    typedef decltype(TENSOR::dummy_scalar()) SCALAR; 

    const GROUP& G; 
    const int n_irreps;

    vector<TENSOR*> parts;

    ~FourierOnGroup(){
      for(auto p:parts) delete p;
    }


  public:

    FourierOnGroup(const GROUP& _G, const cnine::fill_noalloc& fill): 
      G(_G), 
      n_irreps(_G.n_irreps()){
    }

    template<typename FILLTYPE>
    FourierOnGroup(const GROUP& _G, const FILLTYPE& dummy): 
      G(_G), 
      n_irreps(_G.n_irreps()){
      //const int nrho=G.n_irreps;

      for(int i=0; i<n_irreps; i++){
	IRREP rho=G.irrep(i);
	const int d=rho.dim();
	TENSOR* R=new TENSOR(Gdims({d,d}),dummy);
	parts.push_back(R);
      }
    }

    
  public: // Fourier transforms

    template<typename TENSOR2>
    FourierOnGroup(const FunctionOnGroup<GROUP,TENSOR2>& f): 
      G(f.G),
      n_irreps(f.G.n_irreps()){

      const int N=G.size();
      for(int i=0; i<n_irreps; i++){
	IRREP rho=G.irrep(i);
	const int d=rho.dim();
	TENSOR* R=new TENSOR(Gdims({d,d}),cnine::fill_zero());
	for(int j=0; j<N; j++)
	  R->add(rho(j),f(j));
	parts.push_back(R);
      }
    }

    operator FunctionOnGroup<GROUP,TENSOR>() const{
      FunctionOnGroup<GROUP,TENSOR> f(G,cnine::fill_zero());
      const int N=G.size();
      for(int i=0; i<n_irreps; i++){
	IRREP rho=G.irrep(i);
	float c=((float)rho.dim())/N;
	for(int j=0; j<N; j++)
	  f.inc(j,parts[i]->inp(rho(j))*c);
	  }      
      return f;
    }


  public: // Access 

    
  public: // Operations 

    FourierOnGroup left(const ELEMENT& t) const{
      FourierOnGroup R(G,cnine::fill_noalloc());
      for(int i=0; i<n_irreps; i++){
	IRREP rho=G.irrep(i);
	R.parts.push_back(new ctensor(rho(t)*(*parts[i])));
      }
      return R;
    }

    FourierOnGroup right(const ELEMENT& t) const{
      FourierOnGroup R(G,cnine::fill_noalloc());
      for(int i=0; i<n_irreps; i++){
	IRREP rho=G.irrep(i);
	R.parts.push_back(new ctensor((*parts[i])*rho(t)));
      }
      return R;
    }

    FourierOnGroup inv() const{
      FourierOnGroup R(G,cnine::fill_noalloc());
      for(int i=0; i<n_irreps; i++){
	R.parts.push_back(new ctensor(cnine::herm(*parts[i])));
      }
      return R;
    }


  public: // I/O

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<n_irreps; i++)
	oss<<parts[i]->str("  ")<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const FourierOnGroup& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename GROUP, typename TENSOR>
  FourierOnGroup<GROUP,cnine::CtensorObj> Fourier(const FunctionOnGroup<GROUP,TENSOR>& f){
    return  FourierOnGroup<GROUP,cnine::CtensorObj>(f);
  }

  template<typename GROUP>
  FunctionOnGroup<GROUP,cnine::CtensorObj> iFourier(const FourierOnGroup<GROUP,cnine::CtensorObj>& F){
    return  FunctionOnGroup<GROUP,cnine::CtensorObj>(F);
  }

}

#endif
