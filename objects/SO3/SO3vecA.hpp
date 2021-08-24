
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3vecA
#define _SO3vecA

#include "CtensorA.hpp"
#include "SO3partA.hpp"
#include "SO3_CGbank.hpp"
#include "SO3_SPHgen.hpp"


namespace GElib{


  class SO3vecA: public cnine::CtensorA{
  public:


    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;

    class offset: public std::vector<int>{
    public:

      offset(){}

      offset(const SO3type& tau): std::vector<int>(tau.size()){
	int t=0; 
	for(int i=0; i<tau.size(); i++){
	  (*this)[i]=t; 
	  t+=tau[i];
	}
      }

    };


    SO3type tau;
    offset offs;

    
  public: // ---- Constructors -------------------------------------------------------------------------------


    template<typename FILLTYPE>
    SO3vecA(const SO3type& _tau, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      CtensorA({2*_tau.maxl()+1,_tau.total()},_nbu,fill,_dev), tau(_tau), offs(_tau){
    }
    
    SO3vecA(const std::vector<const SO3partA*> v, const int _dev): tau(v.size()){

      const SO3partA* first=nullptr;
      for(auto p:v)
	if(p) {first=p; break;}
      assert(first);
      int nbu=first->get_nbu();

      for(int l=0; l<v.size(); l++)
	if(v[l]) tau[l]=v[l]->getn();
	else tau[l]=0;
      offs=offset(tau);

      SO3vecA R(tau,nbu,cnine::fill::zero,0);
      
      for(int l=0; l<v.size(); l++)
	if(v[l]) R.set_part(l,*v[l]);

      if(dev==0) operator=(std::move(R));
      else operator=(SO3vecA(std::move(R),dev));
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    SO3vecA(const SO3vecA& x): 
      CtensorA(x,cnine::nowarn), tau(x.tau), offs(x.offs){
      CNINE_COPY_WARNING;
    }

    SO3vecA(const SO3vecA& x, const cnine::nowarn_flag& dummy): 
      CtensorA(x,dummy), tau(x.tau), offs(x.offs){}

    SO3vecA(const SO3vecA& x, const int _dev): 
      CtensorA(x,_dev), tau(x.tau), offs(x.offs){}

    SO3vecA(SO3vecA&& x): 
      CtensorA(std::move(x)), tau(x.tau), offs(x.offs){}

    SO3vecA& operator=(const SO3vecA& x){
      CtensorA::operator=(x);
      tau=x.tau;
      offs=x.offs;
      return *this;
    }

    SO3vecA& operator=(const SO3vecA&& x){
      CtensorA::operator=(std::move(x));
      tau=std::move(x.tau);
      offs=std::move(x.offs);
      return *this;
    }

    SO3vecA* clone() const{
      return new SO3vecA(*this, cnine::nowarn_flag());
    }


  public: // ---- Access -------------------------------------------------------------------------------------
    
    
    SO3partA get_part(const int l){
      SO3partA R(l,tau[l],get_nbu(),cnine::fill::zero,0);
      for(int m=0; m<2*l+1; m++)
	for(int i=0; i<tau[l]; i++){
	  R.set_value(m,i,CtensorA::get_value(m,i+offs[l]));
	}
      return R;
    }
    
    void set_part(const int l, const SO3partA& x){
      assert(tau[l]==x.getn());
      assert(get_nbu()==x.get_nbu());

      if(dev==0){
	for(int m=0; m<2*l+1; m++)
	  for(int i=0; i<tau[l]; i++)
	    set_value(m,i+offs[l],x.get(m,i));
      }

      if(dev==1){
	GELIB_UNIMPL();
      }
    }

    void add_part_to(SO3partA& R, const int l){
      assert(R.getl()==l);
      assert(R.getn()==tau[l]);
      for(int m=0; m<2*l+1; m++)
	for(int i=0; i<tau[l]; i++){
	  R.inc(i,m,CtensorA::get_value(m,i+offs[l]));
	}
    }
    
    void add_to_part(const int l, const SO3partA& x){
      assert(tau[l]==x.getn());
      assert(get_nbu()==x.get_nbu());

      if(dev==0){
	for(int m=0; m<2*l+1; m++)
	  for(int i=0; i<tau[l]; i++)
	    inc(m,i+offs[l],x.get(m,i));
      }

      if(dev==1){
	GELIB_UNIMPL();
      }
    }


    void add_to_frags(const SO3type& off, const SO3vecA& x){
      assert(x.tau.size()<=tau.size());
      assert(off.size()==x.tau.size());

      if(dev==0)
	for(int l=0; l<x.tau.size(); l++){
	  assert(tau[l]<=off[l]+x.tau[l]);
	  for(int m=0; m<2*l+1; m++)
	    for(int i=0; i<x.tau[l]; i++)
	      CtensorA::inc(m,i+offs[l]+off[l],x.CtensorA::get_value(m,i+x.offs[l]));
	}

      if(dev==1){
	GELIB_UNIMPL();
      }
    }

    void add_frags_of(const SO3vecA& x, const SO3type& off, const SO3type& w){
      assert(x.tau.size()<=tau.size());
      assert(off.size()==x.tau.size());
      assert(w.size()==x.tau.size());

      if(dev==0)
	for(int l=0; l<x.tau.size(); l++){
	  assert(tau[l]<=w[l]);
	  assert(x.tau[l]<=off[l]+w[l]);
	  for(int m=0; m<2*l+1; m++)
	    for(int i=0; i<w[l]; i++)
	      CtensorA::inc(m,i+offs[l],x.CtensorA::get_value(m,i+x.offs[l]+off[l]));
	}

      if(dev==1){
	GELIB_UNIMPL();
      }
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_prod(const ctensorpack& W, const SO3vecA& x){
      assert(W.fmt==0);
      const int L=tau.size();
      assert(W.tensors.size()==L);
      assert(x.tau.size()==L);
      for(int l=0; l<L; l++){
	if(tau[l]==0) continue;
	assert(W.tensors[l]);
	ctensor& w=*W.tensors[l];
	for(int i=0; i<tau[l]; i++)
	  for(int j=0; j<x.tau[l]; j++)
	    for(int m=0; m<2*l+1; m++)
	      CtensorA::inc(m,i+offs[l],x.CtensorA::get_value(m,j+offs[l])*w.get_value(i,j));
      }
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


#ifdef _WITH_CUDA
    // void CGproduct_cu(const SO3vecA& x, const SO3vecA& y, int maxL, const cudaStream_t& stream);
    // void CGproduct_g1cu(const SO3vecA& xg, const SO3vecA& y, int maxL, const cudaStream_t& stream) const;
    // void CGproduct_g2cu(const SO3vecA& x, const SO3vecA& yg, int maxL, const cudaStream_t& stream) const;
#endif 


    void add_CGproduct(const SO3vecA& x, const SO3vecA& y, int maxL=-1){
      
      if(dev==1){
#ifdef _WITH_CUDA
	assert(x.dev==1);
	assert(y.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	// CGproduct_cu(x,y,maxL,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
	return; 
      }

      if(dev==0){
	
	int L1=x.tau.size()-1; 
	int L2=y.tau.size()-1; 
	int L=tau.size()-1; 
	vector<int> zoffs=offs;

	for(int l1=0; l1<=L1; l1++){
	  const int N1=x.tau[l1];
	  const int xo=x.offs[l1];
	  if(N1==0) continue;
	
	  for(int l2=0; l2<=L2; l2++){
	    const int N2=y.tau[l2];
	    const int yo=y.offs[l2];
	    if(N2==0) continue;

	    for(int l=std::abs(l2-l1); l<=l1+l2 && (maxL<0 || l<=maxL) && l<=L; l++){
	      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	      int zo=zoffs[l];
	      //cout<<l1<<l2<<l<<endl;

	      for(int n1=0; n1<N1; n1++){
		for(int n2=0; n2<N2; n2++)
		  for(int m1=-l1; m1<=l1; m1++)
		    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
		      CtensorA::inc(m1+m2+l,zo+n2,C(m1+l1,m2+l2)*
			x.CtensorA::get_value(m1+l1,xo+n1)*y.CtensorA::get_value(m2+l2,yo+n2));
		zo+=N2;
	      }
	      zoffs[l]+=N1*N2;

	    }
	  }
	}
      }

    }


    void add_CGproduct_back0(const SO3vecA& g, const SO3vecA& y, int maxL=-1){

      if(dev==1){
#ifdef _WITH_CUDA
	assert(g.dev==1);
	assert(y.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	// CGproduct_g1cu(g,y,maxL,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
	return; 
      }

      if(dev==0){
	int L1=tau.size()-1; 
	int L2=y.tau.size()-1; 
	int L=g.tau.size()-1; 
	vector<int> zoffs=g.offs;
	
	for(int l1=0; l1<=L1; l1++){
	  const int N1=tau[l1];
	  const int xo=offs[l1];
	  if(N1==0) continue;
	  
	  for(int l2=0; l2<=L2; l2++){
	    const int N2=y.tau[l2];
	    const int yo=y.offs[l2];
	    if(N2==0) continue;
	
	    for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	      int zo=zoffs[l];

	      for(int n1=0; n1<N1; n1++){
		for(int n2=0; n2<N2; n2++){
		  for(int m1=-l1; m1<=l1; m1++)
		    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		      CtensorA::inc(m1+l1,xo+n1,C(m1+l1,m2+l2)*
			std::conj(y.CtensorA::get_value(m2+l2,yo+n2))*g.CtensorA::get_value(m1+m2+l,zo+n2));
		    }
		}
		zo+=N2;
	      }

	      zoffs[l]+=N1*N2;
	    }
	  }
	}
      }

    }


    void add_CGproduct_back1(const SO3vecA& g, const SO3vecA& x, int maxL=-1){

      if(dev==1){
#ifdef _WITH_CUDA
	assert(x.dev==1);
	assert(g.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	// CGproduct_g2cu(g,x,maxL,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
	return; 
      }

      if(dev==0){
	int L1=x.tau.size()-1; 
	int L2=tau.size()-1; 
	int L=g.tau.size()-1; 
	vector<int> zoffs=g.offs;
      
	for(int l1=0; l1<=L1; l1++){
	  const int N1=x.tau[l1];
	  const int xo=x.offs[l1];
	  if(N1==0) continue;
	  
	  for(int l2=0; l2<=L2; l2++){
	    const int N2=tau[l2];
	    const int yo=offs[l2];
	    if(N2==0) continue;
	
	    for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	      int zo=zoffs[l];

	      for(int n1=0; n1<N1; n1++){
		for(int n2=0; n2<N2; n2++)
		  for(int m1=-l1; m1<=l1; m1++)
		    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
		      inc(m2+l2,yo+n2,C(m1+l1,m2+l2)*
			std::conj(x.CtensorA::get_value(m1+l1,xo+n1))*g.CtensorA::get_value(m1+m2+l,zo+n2));
		zo+=N2;
	      }

	      zoffs[l]+=N1*N2;
	    }
	  }
	}
      }

    }

  };


}

#endif
