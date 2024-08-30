
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3vecB
#define _SO3vecB

#include "GElib_base.hpp"
#include "SO3type.hpp"
#include "SO3partB.hpp"
#include "SO3element.hpp"

#include "SO3_addFFT_Fn.hpp"
#include "SO3_addIFFT_Fn.hpp"

#include "CtensorPackObj.hpp"


namespace GElib{

  #ifdef _WITH_CUDA
  void SO3Fpart_addFproduct_cu(const cnine::Ctensor3_view& r, const cnine::Ctensor3_view& x, 
    const cnine::Ctensor3_view& y, const int conj, const int method, const cudaStream_t& stream);
  //void SO3Fpart_addFproductB_cu(const cnine::Ctensor3_view& r, const cnine::Ctensor3_view& x, 
  //const cnine::Ctensor3_view& y, const int conj, const cudaStream_t& stream);
  #endif




  class SO3vecB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;


    vector<SO3partB*> parts;

    #ifdef WITH_FAKE_GRAD
    SO3vecB* grad=nullptr;
    #endif

    SO3vecB(){}

    ~SO3vecB(){
      for(auto p: parts) delete p;  
      #ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      #endif
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    //SO3vecB(const cnine::fill_noalloc& dummy, const SO3type& _tau, const int _nbu, const int _fmt, const int _dev):
    //tau(_tau), nbu(_nbu), fmt(_fmt), dev(_dev){}


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecB(const int b, const SO3type& tau, const FILLTYPE fill, const int _dev=0){
      for(int l=0; l<tau.size(); l++)
	parts.push_back(new SO3partB(b,l,tau[l],fill,_dev));
    }

    
    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecB(const int b, const int maxl, const FILLTYPE fill, const int _dev=0){
      for(int l=0; l<=maxl; l++)
	parts.push_back(new SO3partB(b,l,2*l+1,fill,_dev));
    }

    
    template<typename... Args>
    SO3vecB(const SO3partB& x0, Args... args){
      SO3vecB(const_variadic_unroller(x0,args...));
    }

    SO3vecB(vector<const SO3partB*> _v){
      for(auto p:_v) parts.push_back(new SO3partB(*p));
    }

    
    // ---- Named constructors --------------------------------------------------------------------------------

    
    static SO3vecB raw(const SO3type& tau, const int _dev=0){
      return SO3vecB(1,tau,cnine::fill_raw(),_dev);}
    static SO3vecB zero(const SO3type& tau, const int _dev=0){
      return SO3vecB(1,tau,cnine::fill_zero(),_dev);}
    static SO3vecB gaussian(const SO3type& tau, const int _dev=0){
      return SO3vecB(1,tau,cnine::fill_gaussian(),_dev);}

    static SO3vecB raw(const int b, const SO3type& tau, const int _dev=0){
      return SO3vecB(b,tau,cnine::fill_raw(),_dev);}
    static SO3vecB zero(const int b, const SO3type& tau, const int _dev=0){
      return SO3vecB(b,tau,cnine::fill_zero(),_dev);}
    static SO3vecB gaussian(const int b, const SO3type& tau, const int _dev=0){
      return SO3vecB(b,tau,cnine::fill_gaussian(),_dev);}
    

    static SO3vecB Fraw(const int maxl){
      return SO3vecB(1,maxl,cnine::fill_raw());}
    static SO3vecB Fzero(const int maxl){
      return SO3vecB(1,maxl,cnine::fill_zero());}
    static SO3vecB Fgaussian(const int maxl){
      return SO3vecB(1,maxl,cnine::fill_gaussian());}

    static SO3vecB Fraw(const int b, const int maxl, const int _dev=0){
      return SO3vecB(b,maxl,cnine::fill_raw(),_dev);}
    static SO3vecB Fzero(const int b, const int maxl, const int _dev=0){
      return SO3vecB(b,maxl,cnine::fill_zero(),_dev);}
    static SO3vecB Fgaussian(const int b, const int maxl, const int _dev=0){
      return SO3vecB(b,maxl,cnine::fill_gaussian(),_dev);}
    

    static SO3vecB raw_like(const SO3vecB& x){
      return SO3vecB::zero(x.getb(),x.get_tau(),x.get_dev());}
    static SO3vecB zeros_like(const SO3vecB& x){
      return SO3vecB::zero(x.getb(),x.get_tau(),x.get_dev());}
    static SO3vecB gaussian_like(const SO3vecB& x){
      return SO3vecB::gaussian(x.getb(),x.get_tau(),x.get_dev());
    }


    // ---- Copying -------------------------------------------------------------------------------------------


    SO3vecB(const SO3vecB& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts.push_back(new SO3partB(*p));
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new SO3vecB(*x.grad);
      #endif 
    }

    SO3vecB(SO3vecB&& x){
      GELIB_MOVE_WARNING();
      parts=x.parts;
      x.parts.clear();
      #ifdef WITH_FAKE_GRAD
      grad=x.grad;
      x.grad=nullptr;
      #endif 
    }


    // ---- Views ---------------------------------------------------------------------------------------------


    SO3vecB view(){
      SO3vecB R;
      int i=0;
      for(auto p: parts){
	R.parts.push_back(new SO3partB(p->CtensorB::view()));
      }
      #ifdef WITH_FAKE_GRAD
      if(grad) R.grad=new SO3vecB(grad->view());
      #endif 
      return R;
    }


    // ---- Transport -----------------------------------------------------------------------------------------


    SO3vecB(const SO3vecB& x, const int _dev){
      for(auto p:x.parts)
	parts.push_back(new SO3partB(p->to_device(_dev)));
    }


    SO3vecB& move_to_device(const int _dev){
      for(auto p:parts)
	p->move_to_device(_dev);
      return *this;
    }

    
    SO3vecB to_device(const int _dev) const{
      SO3vecB R;
      for(auto p:parts)
	R.parts.push_back(new SO3partB(p->to_device(_dev)));
      return R;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    SO3vecB(vector<at::Tensor>& v){
      for(auto& p: v)
	parts.push_back(new SO3partB(p));
    }

    vector<at::Tensor> torch(){
      vector<at::Tensor> R;
      for(auto p: parts)
	R.push_back(p->torch());
      return R;
    }

#endif

 
    // ---- Access --------------------------------------------------------------------------------------------


    int getb() const{
      if(parts.size()>0) return parts[0]->getb();
      return 0;
    }

    SO3type get_tau() const{
      SO3type tau;
      for(auto p:parts)
	tau.push_back(p->getn());
      return tau;
    }

    SO3type get_type() const{
      SO3type tau;
      for(auto p:parts)
	tau.push_back(p->getn());
      return tau;
    }

    int get_maxl() const{
      return parts.size()-1;
    }

    int get_dev() const{
      if(parts.size()>0) return parts[0]->get_dev();
      return 0;
    }

    int get_device() const{
      if(parts.size()>0) return parts[0]->get_dev();
      return 0;
    }

    SO3partB get_part(const int l) const{
      GELIB_CHECK_RANGE(if(l<0||l>=parts.size()) throw std::out_of_range("GElib error: SO3vecB part index "+to_string(l)+" is outside range (0,...,"+to_string(parts.size()-1)+")."));
      return *parts[l];
    }
    
    void forall_parts(std::function<void(const SO3partB& x)> lambda) const{
      int L=parts.size();
      for(int l=0; l<L; l++)
	lambda(*parts[l]);
    }

    void forall_parts(std::function<void(SO3partB& x)> lambda){
      int L=parts.size();
      for(int l=0; l<L; l++)
	lambda(*parts[l]);
    }


    // ---- Cumulative operations ----------------------------------------------------------------------------


    void operator+=(const SO3vecB& x){
      add(x);
    }
    
    void add(const SO3vecB& x){
      assert(parts.size()==x.parts.size());
      for(int l=0; l<parts.size(); l++)
	parts[l]->add(*x.parts[l]);
    }


    // ---- Operations ---------------------------------------------------------------------------------------


    SO3vecB operator+(const SO3vecB& y) const{
      SO3vecB R;
      for(int l=0; l<parts.size(); l++){
	R.parts.push_back(new SO3partB((*parts[l])+(*y.parts[l])));
      }
      return R;
    }


    SO3vecB operator-(const SO3vecB& y) const{
      SO3vecB R;
      for(int l=0; l<parts.size(); l++){
	R.parts.push_back(new SO3partB((*parts[l])-(*y.parts[l])));
      }
      return R;
    }

    SO3vecB operator/(const SO3vecB& y) const{
      SO3vecB R;
      assert(y.parts.size()==parts.size());
      for(int l=0; l<parts.size(); l++){
	R.parts.push_back(new SO3partB( (*parts[l]/(*y.parts[l])) ));
      }
      return R;
    }


    // ---- Products -----------------------------------------------------------------------------------------


    SO3vecB operator*(const cnine::CtensorPackObj& y){
      assert(y.tensors.size()==parts.size());
      SO3type tau;
      for(int l=0; l<parts.size(); l++){
	auto& w=*y.tensors[l];
	assert(w.get_ndims()==2);
	assert(w.get_dim(0)==parts[l]->getn());
	tau.push_back(w.get_dim(1));
      }
      SO3vecB R=SO3vecB::zero(getb(),tau,get_dev());
      R.add_mprod(*this,y);
      return R;
    }


    void add_mprod(const SO3vecB& x, const cnine::CtensorPackObj& y){
      CNINE_DEVICE_SAMEB(x);
      CNINE_DEVICE_SAMEB(y);
      assert(x.parts.size()==y.tensors.size());
      assert(x.parts.size()<=parts.size());
      for(int l=0; l<x.parts.size(); l++)
	parts[l]->add_mprod(*x.parts[l],*y.tensors[l]);
    }


    void add_mprod_back0(const SO3vecB& g, const cnine::CtensorPackObj& y){
      CNINE_DEVICE_SAMEB(g);
      CNINE_DEVICE_SAMEB(y);
      assert(parts.size()==y.tensors.size());
      assert(parts.size()==g.parts.size());
      for(int l=0; l<parts.size(); l++)
	parts[l]->add_mprod_back0(*g.parts[l],*y.tensors[l]);
    }


    void add_mprod_back1_into(cnine::CtensorPackObj& yg, const SO3vecB& x) const{
      CNINE_DEVICE_SAMEB(yg);
      CNINE_DEVICE_SAMEB(x);
      assert(parts.size()==yg.tensors.size());
      assert(parts.size()==x.parts.size());
      for(int l=0; l<parts.size(); l++)
	parts[l]->add_mprod_back1_into(*yg.tensors[l],*x.parts[l]);
    }


    // ---- Rotations ----------------------------------------------------------------------------------------


    SO3vecB rotate(const SO3element& r){
      SO3vecB R;
      for(int l=0; l<parts.size(); l++)
	if(parts[l]) R.parts.push_back(new SO3partB(parts[l]->rotate(r)));
	else R.parts.push_back(nullptr);
      return R;
    }

    
    // ---- CG-products ---------------------------------------------------------------------------------------


    SO3vecB CGproduct(const SO3vecB& y, const int maxl=-1) const{
      assert(getb()==y.getb());

      SO3vecB R=SO3vecB::zero(getb(),GElib::CGproduct(get_tau(),y.get_tau(),maxl),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3vecB& x, const SO3vecB& y){
      assert(get_tau()==GElib::CGproduct(x.get_tau(),y.get_tau(),get_maxl()));

      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l]->add_CGproduct(*x.parts[l1],*y.parts[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }

      
    void add_CGproduct_back0(const SO3vecB& g, const SO3vecB& y){
      assert(g.get_tau()==GElib::CGproduct(get_tau(),y.get_tau(),g.get_maxl()));

      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l1]->add_CGproduct_back0(*g.parts[l],*y.parts[l2],offs[l]);
	    offs[l]+=(parts[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }

      
    void add_CGproduct_back1(const SO3vecB& g, const SO3vecB& x){
      assert(g.get_tau()==GElib::CGproduct(x.get_tau(),get_tau(),g.get_maxl()));

      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l2]->add_CGproduct_back1(*g.parts[l],*x.parts[l1],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(parts[l2]->getn());
	  }
	}
      }
    }

      
    // ---- Blocked CG-products ------------------------------------------------------------------------------


    SO3vecB BlockedCGproduct(const SO3vecB& y, const int bsize, const int maxl=-1) const{
      assert(getb()==y.getb());

      SO3vecB R=SO3vecB::zero(getb(),GElib::BlockedCGproduct(get_tau(),y.get_tau(),bsize,maxl),get_dev());
      R.add_BlockedCGproduct(*this,y,bsize);
      return R;
    }


    void add_BlockedCGproduct(const SO3vecB& x, const SO3vecB& y, const int bsize){
      assert(get_tau()==GElib::BlockedCGproduct(x.get_tau(),y.get_tau(),bsize,get_maxl()));

      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l]->add_BlockedCGproduct(*x.parts[l1],*y.parts[l2],bsize,offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*bsize;
	  }
	}
      }
    }

      
    void add_BlockedCGproduct_back0(const SO3vecB& g, const SO3vecB& y, const int bsize){
      assert(g.get_tau()==GElib::BlockedCGproduct(get_tau(),y.get_tau(),bsize,g.get_maxl()));

      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l1]->add_BlockedCGproduct_back0(*g.parts[l],*y.parts[l2],bsize,offs[l]);
	    offs[l]+=(parts[l1]->getn())*bsize;
	  }
	}
      }
    }

      
    void add_BlockedCGproduct_back1(const SO3vecB& g, const SO3vecB& x, const int bsize){
      assert(g.get_tau()==GElib::BlockedCGproduct(x.get_tau(),get_tau(),bsize,g.get_maxl()));

      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l2]->add_BlockedCGproduct_back1(*g.parts[l],*x.parts[l1],bsize,offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*bsize;
	  }
	}
      }
    }

      
    // ---- Diagonal CG-products -----------------------------------------------------------------------------


    SO3vecB DiagCGproduct(const SO3vecB& y, const int maxl=-1) const{
      return BlockedCGproduct(y,1,maxl);
    }

    void add_DiagCGproduct(const SO3vecB& x, const SO3vecB& y){
      add_BlockedCGproduct(x,y,1);
    }

    void add_DiagCGproduct_back0(const SO3vecB& g, const SO3vecB& y){
      add_BlockedCGproduct_back0(g,y,1);
    }

    void add_DiagCGproduct_back1(const SO3vecB& g, const SO3vecB& x){
      add_BlockedCGproduct_back1(g,x,1);
    }


    // ---- CG-squares ---------------------------------------------------------------------------------------


    SO3vecB CGsquare(const int maxl=-1) const{
      SO3vecB R=SO3vecB::zero(getb(),GElib::CGsquare(get_tau(),maxl),get_dev());
      R.add_CGsquare(*this);
      return R;
    }


    void add_CGsquare(const SO3vecB& x){
      assert(get_tau()==GElib::CGsquare(x.get_tau(),get_maxl()));

      int L1=x.get_maxl(); 
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){

	for(int l=0; l<=2*l1 && l<=L; l++){
	  int n=x.parts[l1]->getn();
	  parts[l]->add_CGsquare(*x.parts[l1],offs[l]);
	  offs[l]+=(n*(n-1))/2+n*(1-l%2);
	}

	for(int l2=l1+1; l2<=L1; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l]->add_CGproduct(*x.parts[l1],*x.parts[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(x.parts[l2]->getn());
	  }
	}
	
      }
    }

      

    // ---- Fproducts ---------------------------------------------------------------------------------------


    SO3vecB Fproduct(const SO3vecB& y, int maxl=-1) const{
      assert(y.getb()==getb());
      if(maxl<0) maxl=get_maxl()+y.get_maxl();
      SO3vecB R=SO3vecB::Fzero(getb(),maxl,get_dev());
      R.add_Fproduct(*this,y);
      return R;
    }


    SO3vecB FproductB(const SO3vecB& y, int maxl=-1) const{
      assert(y.getb()==getb());
      if(maxl<0) maxl=get_maxl()+y.get_maxl();
      SO3vecB R=SO3vecB::Fzero(getb(),maxl,get_dev());
      R.add_FproductB(*this,y);
      return R;
    }


    void add_Fproduct(const SO3vecB& x, const SO3vecB& y, const int method=0){
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    SO3part_addFproductFn(0,method)(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3());
	  }
	}
      }
    }

    
    void add_FproductB(const SO3vecB& x, const SO3vecB& y){
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      if(get_dev()==0){
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    SO3part_addFproductFn()(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3());
	  }
	}
      }
      }

      #ifdef _WITH_CUDA
      if(get_dev()==1){
	for(int l=0; l<=L1+L2 && l<=L ; l++){
	  cudaStream_t stream;
	  CUDA_SAFE(cudaStreamCreate(&stream));
	  for(int l1=std::max(0,l-L2); l1<=std::min(l+L2,L1); l1++){
	    for(int l2=std::abs(l-l1); l2<=std::min(l+l1,L2); l2++){
	      SO3Fpart_addFproduct_cu(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3(),0,0,stream);
	    }
	  }
	  CUDA_SAFE(cudaStreamSynchronize(stream));
	  CUDA_SAFE(cudaStreamDestroy(stream));
	}
      }
      #endif 
    }


    void add_Fproduct_back0(const SO3vecB& g, const SO3vecB& y, const int method=0){
      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back0Fn(0,method)(parts[l1]->view3(),g.parts[l]->view3(),y.parts[l2]->view3());
	  }
	}
      }
    }


    void add_Fproduct_back1(const SO3vecB& g, const SO3vecB& x){
      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back1Fn()(parts[l2]->view3(),g.parts[l]->view3(),x.parts[l1]->view3());
	  }
	}
      }
    }

    
    // ---- Fmodsq -------------------------------------------------------------------------------------------
    

    SO3vecB Fmodsq(int maxl=-1) const{
      if(maxl<0) maxl=2*get_maxl();
      SO3vecB R=SO3vecB::Fzero(getb(),maxl,get_dev());
      R.add_Fmodsq(*this,*this);
      return R;
    }


    void add_Fmodsq(const SO3vecB& x, const SO3vecB& y){
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    SO3part_addFproductFn(1)(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3().flip());
	  }
	}
      }
    }

    void add_Fmodsq_back(const SO3vecB& g, const SO3vecB& x){
      add_Fmodsq_back0(g,x);
      add_Fmodsq_back1(g,x);
    }

    void add_Fmodsq_back0(const SO3vecB& g, const SO3vecB& y){
      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back0Fn(1)(parts[l1]->view3(),g.parts[l]->view3(),y.parts[l2]->view3().flip());
	  }
	}
      }
    }

    void add_Fmodsq_back1(const SO3vecB& g, const SO3vecB& x){
      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back1Fn(1)(parts[l2]->view3().flip(),g.parts[l]->view3(),x.parts[l1]->view3());
	  }
	}
      }
    }


  public: // ---- FFT ---------------------------------------------------------------------------------------


    cnine::CtensorB iFFT(const int n0, const int n1, const int n2) const{
      cnine::CtensorB R=cnine::CtensorB::zero(cnine::Gdims(getb(),n0,n1,n2),get_dev());
      add_iFFT_to(R);
      return R;
    }

    void add_iFFT_to(cnine::CtensorB& R) const{
      forall_parts([&](const SO3partB& x){
	  SO3part_addIFFT_Fn()(R.view4(),x.view3());
	});
    }

    void add_FFT(const cnine::CtensorB& R){
      forall_parts([&](const SO3partB& x){
	  SO3part_addFFT_Fn()(x.view3(),R.view4());
	});
    }


  public: // ---- Experimental -------------------------------------------------------------------------------



    #ifdef WITH_FAKE_GRAD
    void add_to_grad(const SO3vecB& x){
      if(grad) grad->add(x);
      else grad=new SO3vecB(x);
    }

    void add_to_part_of_grad(const int l, const SO3partB& x){
      if(!grad) grad=new SO3vecB(SO3vecB::zeros_like(*this));
      grad->parts[l]->add(x);
    }

    SO3vecB& get_grad(){
      if(!grad) grad=new SO3vecB(SO3vecB::zeros_like(*this));
      return *grad;
    }

    SO3vecB view_of_grad(){
      cout<<"view"<<endl;
      if(!grad) grad=new SO3vecB(SO3vecB::zeros_like(*this));
      return grad->view();
    }
    #endif 


  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::SO3vecB";
    }

    string str(const string indent="") const{
      ostringstream oss;
	for(int l=0; l<parts.size(); l++){
	  if(!parts[l]) continue;
	  oss<<indent<<"Part l="<<l<<":\n";
	  oss<<parts[l]->str(indent+"  ");
	  oss<<endl;
	}
      return oss.str();
    }

    string repr(const string indent="") const{
      return "<GElib::SO3vecB of type "+get_tau().str()+" [b="+to_string(getb())+"]>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vecB& x){
      stream<<x.str(); return stream;
    }

  };


  // ---- Post-class functions -------------------------------------------------------------------------------


  //inline std::vector<SO3type> get_types(const std::vector<const SO3vecB*>& v){
  //vector<SO3type> R;
  //for(auto p:v)
  //  R.push_back(p->get_type());
  //return R;
  //}


  

  // ---- Stand-alone functions ------------------------------------------------------------------------------


  inline cnine::CtensorB SO3_iFFT(const SO3vecB& v, const int n0, const int n1, const int n2){
    return v.iFFT(n0,n1,n2);
  }

  inline SO3vecB SO3_FFT(const cnine::CtensorB& f, const int _maxl){
    assert(f.ndims()==4);
    int B=f.dim(0);
    SO3vecB R=SO3vecB::Fzero(B,_maxl,f.get_dev());
    R.add_FFT(f);
    return R;
  }

}

#endif
