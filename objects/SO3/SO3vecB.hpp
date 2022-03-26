
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


namespace GElib{


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


    SO3vecB(){}

    ~SO3vecB(){
      for(auto p: parts) delete p;  
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
      for(auto& p:x.parts)
	parts.push_back(p);
    }

    SO3vecB(SO3vecB&& x){
      parts=x.parts;
      x.parts.clear();
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


    void add_Fproduct(const SO3vecB& x, const SO3vecB& y){
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    SO3part_addFproduct_Fn()(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3());
	  }
	}
      }
    }


    void add_Fproduct_back0(const SO3vecB& g, const SO3vecB& y){
      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back0Fn()(parts[l1]->view3(),g.parts[l]->view3(),y.parts[l2]->view3());
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
	    SO3part_addFproduct_Fn(1)(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3().flip());
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


  public: // ---- I/O ---------------------------------------------------------------------------------------


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
      return "<GElib::SO3vecB of type"+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vecB& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif
