
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3vecObj
#define _SO3vecObj

#include "GElib_base.hpp"
//#include "Dobject.hpp"
#include "SO3type.hpp"
#include "SO3part.hpp"
#include "SO3vecA.hpp"
#include "SO3element.hpp"
#include "CtensorPackObj.hpp"
//#include "SO3vecObj_helpers.hpp"
//#include "GenericOperators.hpp"


namespace GElib{


  class SO3vec{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;

    typedef GELIB_SO3VEC_IMPL SO3veci;


    SO3type tau; 
    int nbu; 
    int dev=0; 
    vector<SO3part*> parts;
    SO3veci* vec=nullptr;
    
    int fmt=0;

    SO3vec(){}

    ~SO3vec(){
      for(auto p: parts) delete p;  
      delete vec;
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    SO3vec(const cnine::fill_noalloc& dummy, const SO3type& _tau, const int _nbu, const int _fmt, const int _dev):
      tau(_tau), nbu(_nbu), fmt(_fmt), dev(_dev){}


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const int _fmt, const int _dev): 
      tau(_tau), nbu(_nbu), dev(_dev), fmt(_fmt){
      if(fmt==0){
	for(int l=0; l<tau.size(); l++)
	  if(tau[l]>0) parts.push_back(new SO3part(l,tau[l],nbu,fill,_dev));
	  else parts.push_back(nullptr);
      }
      if(fmt==1){
	vec=new SO3veci(_tau,_nbu,fill,_dev);
      }
    }


    // ---- without nbu

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const FILLTYPE fill): 
      SO3vec(_tau,-1,fill,0,0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const FILLTYPE fill, const SO3vec_format& _format): 
      SO3vec(_tau,-1,fill,toint(_format),0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const FILLTYPE fill, const device& _device): 
      SO3vec(_tau,-1,fill,0,_device.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const FILLTYPE fill, const SO3vec_format& _format, const device& _device): 
      SO3vec(_tau,-1,fill,toint(_format),_device.id()){}


    // ---- with nbu

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill): 
      SO3vec(_tau,_nbu,fill,0,0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const SO3vec_format& _format): 
      SO3vec(_tau,_nbu,fill,toint(_format),0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const device& _device): 
      SO3vec(_tau,_nbu,fill,0,_device.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const SO3vec_format& _format, 
      const device& _device): 
      SO3vec(_tau,_nbu,fill,toint(_format),_device.id()){}


    static SO3vec zero(const SO3type& tau){
      return SO3vec(tau,-1,cnine::fill::zero);}
    static SO3vec zero(const SO3type& tau, const SO3vec_format& _format){
      return SO3vec(tau,-1,cnine::fill::zero,_format);}
    static SO3vec zero(const SO3type& tau, const device& _device){
      return SO3vec(tau,-1,cnine::fill::zero,_device);}
    static SO3vec zero(const SO3type& tau, const SO3vec_format& _format, const device& _device){
      return SO3vec(tau,-1,cnine::fill::zero,_format,_device);}
    
    static SO3vec ones(const SO3type& tau){
      return SO3vec(tau,-1,cnine::fill::ones);}
    static SO3vec ones(const SO3type& tau, const SO3vec_format& _format){
      return SO3vec(tau,-1,cnine::fill::ones,_format);}
    static SO3vec ones(const SO3type& tau, const device& _device){
      return SO3vec(tau,-1,cnine::fill::ones,_device);}
    static SO3vec ones(const SO3type& tau, const SO3vec_format& _format, const device& _device){
      return SO3vec(tau,-1,cnine::fill::ones,_format,_device);}
    
    static SO3vec gaussian(const SO3type& tau){
      return SO3vec(tau,-1,cnine::fill::gaussian);}
    static SO3vec gaussian(const SO3type& tau, const SO3vec_format& _format){
      return SO3vec(tau,-1,cnine::fill::gaussian,_format);}
    static SO3vec gaussian(const SO3type& tau, const device& _device){
      return SO3vec(tau,-1,cnine::fill::gaussian,_device);}
    static SO3vec gaussian(const SO3type& tau, const SO3vec_format& _format, const device& _device){
      return SO3vec(tau,-1,cnine::fill::gaussian,_format,_device);}
    


  public: // ---- Construction from parts ---------------------------------------------------------------------


    template<typename... Args>
    SO3vec(const SO3part& x0, const SO3part& x1, Args...args){
      vector<SO3part*> argv;
      const_parts_unroller_sub(argv, x0, x1, args...);
      (*this)=SO3vec(argv,dev); // problem with this?
      //(*this)=std::move(SO3vec(argv,dev)); // problem with this?
    }

  private:

    template<typename... Args>
    void const_parts_unroller_sub(vector<SO3part*>& argv, const SO3part& x, Args&... args){
      argv.push_back(const_cast<SO3part*>(&x));
      const_parts_unroller_sub(argv, args...);
    }

    void const_parts_unroller_sub(vector<SO3part*>& argv, const SO3part& x){
      argv.push_back(const_cast<SO3part*>(&x));
    }

    void const_parts_unroller_sub(vector<SO3part*>& argv, const SO3vec_format& _format){
      fmt=toint(_format);
    }

    SO3vec(const vector<SO3part*>& v, const device& _dev=0): 
      dev(_dev.id()){
      SO3part* first; for(auto p: v) if(p) {first=p; break;}
      if(!first) return;
      nbu=first->get_nbu();

      for(int l=0; l<v.size(); l++){
	if(!v[l]){
	  tau.push_back(0);
	  parts.push_back(nullptr);
	  continue;
	}
	const SO3part& P=*v[l];
	//GENET_CHECK_NBU2(nbu,P.get_nbu());
	assert(P.getl()==l);
	tau.push_back(P.getn());
	parts.push_back(new SO3part(P,dev));
      }
    }

    
  public: // ---- Construction by concatenation --------------------------------------------------------------


    /*
    SO3vec(const vector<const SO3vec*> v){
      const int N=v.size();
      assert(N>0);
      nbu=v[0]->nbu;
      tau=v[0]->tau;

      for(int i=1; i<N; i++){
	GELIB_CHECK_SO3VEC_FORMAT_IS_0(v[i]->fmt);
	tau+=v[i]->tau;
      }

      for(int l=0; l<v[0]->parts.size(); l++){
	if(tau[l]>0){
	  vector<const SO3part*> w;
	  for(int i=0; i<N; i++) w.push_back(v[i]->parts[l]);
	  parts.push_back(new SO3part(w));
	}else 
	  parts.push_back(nullptr);
      }
    }
    */

      
   public: // ---- Copying ------------------------------------------------------------------------------------
    

    SO3vec(const SO3vec& x):
      tau(x.tau), nbu(x.nbu), dev(x.dev), fmt(x.fmt){
      CNINE_COPY_WARNING();
      for(auto p: x.parts)
	if(p) parts.push_back(new SO3part(*p));
	else parts.push_back(nullptr);
      if(vec) vec=x.vec->clone();
    }

    SO3vec(const SO3vec& x, const cnine::nowarn_flag& dummy):
      tau(x.tau), nbu(x.nbu), dev(x.dev), fmt(x.fmt){
      for(auto p: x.parts)
	if(p) parts.push_back(new SO3part(*p));
	else parts.push_back(nullptr);
      if(x.vec) vec=x.vec->clone();
    }
      
    SO3vec(SO3vec&& x):
      tau(x.tau), nbu(x.nbu), dev(x.dev), fmt(x.fmt){
      for(auto p: x.parts)
	parts.push_back(p);
      x.parts.clear(); 
      delete vec;
      vec=x.vec;
    }

    SO3vec& operator=(const SO3vec& x){
      tau=x.tau; 
      nbu=x.nbu; 
      dev=x.dev;
      fmt=x.fmt;
      for(auto p: parts) delete p;
      parts.clear(); 
      for(auto p: x.parts)
	if(p) parts.push_back(new SO3part(*p));
	else parts.push_back(nullptr);
      if(x.vec) vec=x.vec->clone(); //new SO3veci(*x.vec);
     return *this;
    }

    SO3vec& operator=(SO3vec&& x){
      tau=x.tau; 
      nbu=x.nbu; 
      dev=x.dev;
      fmt=x.fmt;
      for(auto p: parts) delete p;
      parts.clear(); 
      for(auto p: x.parts)
	parts.push_back(p);
      x.parts.clear();
      vec=x.vec;
      x.vec=nullptr;
      return *this;
    }

    SO3vec(const SO3vec& x, const cnine::fill_zero& fill):
      SO3vec(x.tau,x.nbu,fill,x.fmt,x.dev){}

    
  public: // ---- Variants ----------------------------------------------------------------------------------


    SO3vec(const SO3vec& x, const device& _dev):
      SO3vec(cnine::fill::noalloc,x.tau,x.nbu,x.fmt,_dev.id()){
      if(fmt==0){
	for(auto p: x.parts)
	  if(p) parts.push_back(new SO3part(*p,_dev));
	  else parts.push_back(nullptr);
      }
      if(fmt==1) vec=new SO3veci(*x.vec,_dev.id());
    }
    
    /*
    SO3vec(const SO3vec& x, const int _dev=0):
      SO3vec(cnine::fill::noalloc,x.tau,x.nbu,x.fmt,_dev){
      if(fmt==0){
	for(auto p: x.parts)
	  if(p) parts.push_back(new SO3part(*p,_dev));
	  else parts.push_back(nullptr);
      }
      if(fmt==1) vec=new SO3veci(*x.vec,_dev);
    }
    */

    SO3vec(const SO3vec& x, const SO3vec_format& _fmt):
      SO3vec(cnine::fill::noalloc,x.tau,x.nbu,toint(_fmt),x.dev){
      
      if(x.fmt==fmt){
	(*this)=SO3vec(x,cnine::nowarn);
	return;
      }
    
      if(fmt==1 && x.fmt==0){
	std::vector<const GELIB_SO3PART_IMPL*> v(x.parts.size());
	for(int l=0; l<x.parts.size(); l++)
	  if(x.parts[l]) v[l]=x.parts[l];
	  else v[l]=nullptr;
	vec=new SO3veci(v,dev);
      }

      if(fmt==0 && x.fmt==1){
	for(int l=0; l<tau.size(); l++)
	  if(tau[l]>0) parts.push_back(new SO3part(x.vec->get_part(l)));
	  else parts.push_back(nullptr);
      }

    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    SO3vec to(const device& _dev) const{
      return SO3vec(*this,_dev);
    }

    SO3vec to_device(const int _dev) const{
      return SO3vec(*this,_dev);
    }

    SO3vec to(const SO3vec_format& _fmt) const{
      if(fmt==toint(_fmt)) return *this;
      return SO3vec(*this,_fmt);
    }

    SO3vec to_format(const int _fmt) const{
      if(fmt==_fmt) return *this;
      if(_fmt==1) return SO3vec(*this,SO3vec_format::compact);
      return SO3vec(*this,SO3vec_format::parts);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------

    
    SO3type get_tau() const{
      return tau;
    }

    SO3type type() const{
      return tau;
    }

    int get_nbu() const{
      return nbu; 
    }

    int get_dev() const{
      return dev; 
    }

    int getL() const{
      return tau.size()-1; 
    }

    int nparts() const{
      return tau.size(); 
    }

    int size() const{
      return tau.size(); 
    }

    int get_device() const{
      return dev;
    }

    int get_format() const{
      return fmt; 
    }

    
    // ---- parts -------------------------------------


    SO3part get_part(const int l) const{
      if(fmt==0){
	if(parts[l]) return SO3part(*parts[l]);
	else return SO3part(l,0,nbu,dev);
      }
      if(fmt==1){
	return SO3part(vec->get_part(l));
      }
      return *parts[l];
    }

    SO3vec& set_part(const int l, const SO3part& x){
      //GENET_CHECK_NBU2(nbu,x.nbu);
      assert(l==x.getl());

      const int maxl=tau.maxl();
      if(l>maxl) tau.resize(l+1,0);

      if(fmt==0){
	if(l>maxl)
	  parts.resize(l+1,nullptr);
	if(parts[l]) delete parts[l];
	tau[l]=x.getn();
	parts[l]=new SO3part(x);
      }

      if(fmt==1){
	vec->set_part(l,x);
      }

      return *this;
    }


    // ---- SO3vec_part

    /*
    SO3vec_part operator[](const int l){
      assert(fmt==0);
      return SO3vec_part(*parts[l],this);
    }
    */


    SO3part part(const int l){
      assert(fmt==0);
      assert(l<parts.size());
      return SO3part(*parts[l],cnine::view_flag());
    }

    const SO3part* partp(const int l) const{ // for compatibility with broadcast ops
      assert(fmt==0);
      assert(l<parts.size());
      return parts[l];
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void set_zero(){
      if(fmt==0)
	for(auto p:parts)
	  p->set_zero();
      if(fmt==2)
	vec->set_zero();
    }

    void norm2_into(cscalar& R) const{
      if(fmt==0)
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_norm2_into(R);
      if(fmt==1)
	vec->add_norm2_into(R);
    }

    void add_inp_into(cscalar& R, const SO3vec& y) const{
      assert(y.nparts()==nparts());
      if(fmt==0)
	for(int l=0; l<parts.size(); l++){
	  if(parts[l] && y.parts[l]) parts[l]->add_inp_into(R,*y.parts[l]);
	}
      if(fmt==1){
	assert(y.fmt==1);
	vec->add_inp_into(R,*y.vec);
      }
    }

    /*
    void add_inp_into(CtensorPackObj& R, const SO3vec& y) const{

      if(fmt==0){
	assert(y.nparts()==nparts());
	for(int l=0; l<parts.size(); l++)
	  if(parts[l] && y.parts[l]) R.T[l]->add_Mprod(*y.parts[l],*parts[l]);
      }

      if(fmt==1){
	for(int l=0; l<tau.size(); l++){
	  //if(tau[l]>0 && y.tau[l]>0) replace(R.T[l]->hdl,Cengine_engine->push<SO3vec_add_Minp_part_op>
	  //(R.T[l]->hdl,hdl,y.hdl,parts[l]->dims,y.parts[l]->dims,l));
	}
      }

      if(fmt==2){
	FCG_UNIMPL();
      }

    }
    */

    void normalize_fragments() const{
      GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);

      if(fmt==0){
	//for(auto p: parts)
	//if(p) p->normalize_fragments();
      }

    }


  public: // ---- Accumulating operations ------------------------------------------------------------------


    void add(const SO3vec& x){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	assert(x.nparts()==nparts());
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add(*x.parts[l]);
      }

      if(fmt==1)
	vec->add(*x.vec);
    }

    void add(const SO3vec& x, const float c){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	assert(x.nparts()==nparts());
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add(*x.parts[l],c);
      }

      if(fmt==1)
	vec->add(*x.vec,c);
    }

    void add(const SO3vec& x, const complex<float> c){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	assert(x.nparts()==nparts());
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add(*x.parts[l],c);
      }

      if(fmt==1)
	vec->add(*x.vec,c);
    }

    void add(const SO3vec& x, const rscalar& c){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add(*x.parts[l],c);
      }

      if(fmt==1)
	vec->add(*x.vec,c);
    }

    void add(const SO3vec& x, const cscalar& c){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add(*x.parts[l],c);
      }

      if(fmt==1)
	vec->add(*x.vec,c);
    }

    void add_cconj(const SO3vec& x, const cscalar& c){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_cconj(*x.parts[l],c);
      }

      if(fmt==1)
	vec->add_cconj(*x.vec,c);
    }

    void add_conj(const SO3vec& x){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_conj(*x.parts[l]);
      }

      if(fmt==1)
	vec->add_conj(*x.vec);
    }

    void add_conj(const SO3vec& x, const cscalar& c){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_conj(*x.parts[l],c);
      }

      if(fmt==1)
	vec->add_conj(*x.vec);
    }

    void subtract(const SO3vec& x){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->subtract(*x.parts[l]);
      }

      if(fmt==1)
	vec->subtract(*x.vec);
    }

    void add_prod(const ctensorpack& W, const SO3vec& x){
      GELIB_CHECK_NBU3(nbu,W.nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0){
	assert(x.nparts()>=nparts());
	assert(W.tensors.size()==nparts());
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_prod(*W.tensors[l],*x.parts[l]);      
      }

      if(fmt==1){
	vec->add_prod(W,*x.vec);
      }

    }


    /*
    void add_prod_back0_into(CtensorPackObj& Wg, const SO3vec& x) const{
      if(fmt==0){
	assert(x.nparts()>=nparts());
	assert(Wg.T.size()==nparts());
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_prod_back0_into(*Wg.T[l],*x.parts[l]);
      }
      if(fmt==1){
	assert(x.tau.size()>=tau.size());
	assert(Wg.T.size()==tau.size());
	for(int l=0; l<tau.size(); l++)
	  if(tau[l]>0) replace(Wg.T[l]->hdl,Cengine_engine->push<SO3vec_add_Mprod_part_op<1> >
	    (Wg.T[l]->hdl,hdl,x.hdl,Wg.T[l]->dims,Gdims(2*l+1,tau[l]),l));
      }
      //if(fmt==1) replace(Wg.hdl,Cengine_engine->push<SO3vec_add_prod_back0>(Wg.hdl,hdl,x.hdl));
    }

    void add_prod_back1_into(const CtensorPackObj& W, SO3vec& xg) const{
      if(fmt==0){
	assert(xg.nparts()>=nparts());
	assert(W.T.size()==nparts());
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_prod_back1_into(*W.T[l],*xg.parts[l]);
      }
      if(fmt==1){
	assert(xg.tau.size()>=tau.size());
	assert(W.T.size()==tau.size());
	for(int l=0; l<tau.size(); l++)
	  if(tau[l]>0) replace(xg.hdl,Cengine_engine->push<SO3vec_add_Mprod_part_op<2> >
	    (xg.hdl,hdl,W.T[l]->hdl,Gdims(2*l+1,tau[l]),W.T[l]->dims,l));
      }
      //if(fmt==1) replace(xg.hdl,Cengine_engine->push<SO3vec_add_prod_back0>(xg.hdl,hdl,W.hdl));
    }
    */


    void add_plus(const SO3vec& x, const SO3vec& y){
      GELIB_CHECK_TAU3(tau,x.tau,y.tau);
      GELIB_CHECK_NBU3(nbu,x.nbu,y.nbu);
      GELIB_CHECK_SO3FORMAT3(fmt,x.fmt,y.fmt);

      if(fmt==0)
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_plus(*x.parts[l],*y.parts[l]);
      if(fmt==1)
	vec->add_plus(*x.vec,*y.vec);
    }

    void add_minus(const SO3vec& x, const SO3vec& y){
      GELIB_CHECK_TAU3(tau,x.tau,y.tau);
      GELIB_CHECK_NBU3(nbu,x.nbu,y.nbu);
      GELIB_CHECK_SO3FORMAT3(fmt,x.fmt,y.fmt);

      if(fmt==0)
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_minus(*x.parts[l],*y.parts[l]);
      if(fmt==1)
	vec->add_minus(*x.vec,*y.vec);
    }

    void add_to_part(const SO3part& x, const int l){
      if(fmt==0){
	assert(parts[l]);
	parts[l]->add(x);
      }

      if(fmt==1)
	vec->add_to_part(l,x);
    }

    void add_part_to(SO3part& x, const int l) const{
      if(fmt==0){
	assert(parts[l]);
	x.add(*parts[l]);
      }

      if(fmt==1)
	vec->add_part_to(x,l);
    }
 
    void add_norm2_back(const cscalar& g, const SO3vec& x){
      GELIB_CHECK_TAU2(tau,x.tau);
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0)
	for(int l=0; l<tau.size(); l++)
	  if(parts[l]){
	    parts[l]->add(*x.parts[l],g);
	    parts[l]->add_conj(*x.parts[l],g);
	  }

      if(fmt==1){
	vec->add(*x.vec,g);
	vec->add_conj(*x.vec,g);
      }
    }

    void add_to_frags(const SO3type& off, const SO3vec& x){
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);

      if(fmt==0)
	for(int l=0; l<parts.size(); l++)
	  if(parts[l]) parts[l]->add_to_frags(off[l],*x.parts[l]);
 
      if(fmt==1)
	vec->add_to_frags(off,*x.vec);
    }

    void add_frags_of(const SO3vec& x, const SO3type& off, const SO3type& w){
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);
      assert(x.tau.size()<=tau.size());
      assert(off.size()==x.tau.size());
      assert(w.size()==x.tau.size());

      if(fmt==0)
	for(int l=0; l<x.parts.size(); l++)
	  if(parts[l]) parts[l]->add_frags_of(*x.parts[l],off[l],w[l]);
 
      if(fmt==1)
	vec->add_frags_of(*x.vec,off,w);
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    void add_CGproduct(const SO3vec& x, const SO3vec& y, const int maxL=-1){
      GELIB_CHECK_NBU3(nbu,x.nbu,y.nbu);
      GELIB_CHECK_SO3FORMAT3(fmt,x.fmt,y.fmt);
      assert(tau==GElib::CGproduct(x.tau,y.tau,maxL));

      if(fmt==0){
	int L1=x.getL(); 
	int L2=y.getL();
	vector<int> offs(tau.size(),0);
	
	for(int l1=0; l1<=L1; l1++){
	  if(x.tau[l1]==0) continue;
	  for(int l2=0; l2<=L2; l2++){
	    if(y.tau[l2]==0) continue;
	    for(int l=std::abs(l2-l1); l<=l1+l2 && (maxL<0 || l<=maxL); l++){
	      parts[l]->add_CGproduct(*x.parts[l1],*y.parts[l2],offs[l]);
	      offs[l]+=(x.parts[l1]->getn())*(y.parts[l2]->getn());
	    }
	  }
	}
      }

      if(fmt==1) 
	vec->add_CGproduct(*x.vec,*y.vec,maxL);
    }


    void add_CGproduct_back0(const SO3vec& g, const SO3vec& y, const int maxL=-1){
      GELIB_CHECK_NBU3(nbu,g.nbu,y.nbu);
      GELIB_CHECK_SO3FORMAT3(fmt,g.fmt,y.fmt);

      if(fmt==0){
	int L1=getL(); 
	int L2=y.getL();
	int L=g.getL();
	vector<int> offs(g.tau.size(),0);
	
	for(int l1=0; l1<=L1; l1++){
	  if(tau[l1]==0) continue;
	  for(int l2=0; l2<=L2; l2++){
	    if(y.tau[l2]==0) continue;
	    for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	      parts[l1]->add_CGproduct_back0(*g.parts[l],*y.parts[l2],offs[l]);
	      offs[l]+=(this->parts[l1]->getn())*(y.parts[l2]->getn());
	    }
	  }
	}
      }

      if(fmt==1) 
	vec->add_CGproduct_back0(*g.vec,*y.vec,maxL);

    }


    void add_CGproduct_back1(const SO3vec& g, const SO3vec& x, const int maxL=-1){
      GELIB_CHECK_NBU3(nbu,x.nbu,g.nbu);
      GELIB_CHECK_SO3FORMAT3(fmt,x.fmt,g.fmt);

      if(fmt==0){
	int L1=x.getL(); 
	int L2=getL();
	int L=g.getL();
	vector<int> offs(g.tau.size(),0);
	
	for(int l1=0; l1<=L1; l1++){
	  if(x.tau[l1]==0) continue;
	  for(int l2=0; l2<=L2; l2++){
	    if(tau[l2]==0) continue;
	    for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	      parts[l2]->add_CGproduct_back1(*g.parts[l],*x.parts[l1],offs[l]);
	      offs[l]+=(x.parts[l1]->getn())*(this->parts[l2]->getn());
	    }
	  }
	}
      }

      if(fmt==1) 
	vec->add_CGproduct_back1(*g.vec,*x.vec,maxL);

    }

    
  public: // ---- Normalization ------------------------------------------------------------------------------

    /*
    SO3vec normalize() const{
      SO3vec R;
      R.tau=tau;
      R.nbu=nbu;
      for(int l=0; l<parts.size(); l++)
	R.parts.push_back(new SO3part(parts[l]->normalize()));
      return R;      
    }

    void add_normalize_back(const SO3vec& g, const SO3vec& x){
      assert(x.nparts()==nparts());
      assert(g.nparts()==nparts());
      for(int l=0; l<parts.size(); l++)
	parts[l]->add_normalize_back(*g.parts[l],*x.parts[l]);
    }
    */

  public: // ---- Not in-place operations --------------------------------------------------------------------


    SO3vec plus(const SO3vec& x){
      GELIB_CHECK_NBU2(nbu,x.nbu);
      GELIB_CHECK_SO3FORMAT2(fmt,x.fmt);
      GELIB_CHECK_TAU2(tau,x.tau);
      assert(x.nparts()==nparts());

      if(fmt==0){
	SO3vec R;
	R.tau=tau;
	R.nbu=nbu;
	R.fmt=fmt;
	for(int l=0; l<parts.size(); l++)
	  if(parts[l] && x.parts[l]) R.parts.push_back(new SO3part(parts[l]->plus(*x.parts[l])));
	  else R.parts.push_back(nullptr);
	return R;
      }

      if(fmt==1){
	GELIB_UNIMPL();
      }
	
      return SO3vec();
    }

    /*
    SO3vec chunk(const SO3type& offs, const SO3type& width) const{
      GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);
      assert(offs.size()==width.size());
      assert(offs.size()<=parts.size());

      SO3vec R(fill::noalloc,width,nbu,fmt,dev);
      for(int l=0; l<offs.size(); l++)
	if(parts[l]) R.parts.push_back(new SO3part(parts[l]->chunk(offs[l],width[l])));
	else R.parts.push_back(nullptr);
      return R;
    }
    */
  
    SO3vec rotate(const SO3element& r){
      //GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);
      if(fmt!=0) return to_format(0).rotate(r).to_format(fmt);

      SO3vec R(cnine::fill::noalloc,tau,nbu,0,dev);
      for(int l=0; l<parts.size(); l++)
	if(parts[l]) R.parts.push_back(new SO3part(parts[l]->rotate(r)));
	else R.parts.push_back(nullptr);
      return R;
    }
    
    /*
    CtensorPackObj fragment_norms() const{
      assert(parts.size()>0);
      CtensorPackObj P;
      for(auto p: parts)
	P.push_back(new ctensor());
      return P;
    }
    */
  

  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    /*
    SO3part static spharm(const SO3type& _tau, const ctensor& x, const int _nbu=-1, const device_id& dev=0){
      SO3vec v;
      v.tau=_tau; 
      v.nbu=_nbu;
      v.parts.resize(tau.size());
      for(int l=0; l<tau.size(); l++)
	v.parts[l]=new SO3part(engine::new_SO3part_spharm(l,x,nbu,dev),l,1,nbu);
      return v;
    }
    */

    /*
    SO3vec static spharm(const SO3type& _tau, const Gtensor<float>& x, const int _nbu=-1, const device_id& dev=0){
      assert(x.k==1);
      assert(x.dims[0]==3);
      SO3vec v;
      v.tau=_tau; 
      v.nbu=_nbu;
      v.parts.resize(_tau.size());
      for(int l=0; l<_tau.size(); l++)
	if(_tau[l]>0) v.parts[l]=SO3part::spharm(l,_tau[l],x,_nbu,dev);
	else v.parts[l]=nullptr;
      return v;
    }

    
    SO3vec static spharm(const SO3type& _tau, const float x, const float y, const float z, const int _nbu=-1, const device_id& dev=0){
      SO3vec v;
      v.tau=_tau; 
      v.nbu=_nbu;
      v.parts.resize(_tau.size());
      for(int l=0; l<_tau.size(); l++)
	if(_tau[l]>0) v.parts[l]=new SO3part(SO3part::spharm(l,_tau[l],x,y,z,_nbu,dev));
	else v.parts[l]=nullptr;
      return v;
    }
    */


  // ---- In-place operators ---------------------------------------------------------------------------------


    SO3vec& operator+=(const SO3vec& y){
      add(y);
      return *this;
    }
    
    SO3vec& operator-=(const SO3vec& y){
      subtract(y);
      return *this;
    }


    // ---- Binary operators ---------------------------------------------------------------------------------


    SO3vec operator+(const SO3vec& y) const{
      SO3vec R(*this,cnine::nowarn);
      R.add(y);
      return R;
    }

    SO3vec operator-(const SO3vec& y) const{
      SO3vec R(*this,cnine::nowarn);
      R.subtract(y);
      return R;
    }
    
    SO3vec operator*(const cscalar& c) const{
      SO3vec R(get_tau(),nbu,cnine::fill::zero);
      R.add(*this,c);
      return R;
    }

    SO3vec times(const ctensorpack& W) const{
      cout<<"fmt"<<fmt<<endl;
      SO3vec R(SO3type::left(W),nbu,cnine::fill::zero,fmt,dev);
      R.add_prod(W,*this);
      return R;
    }

    /*
    CtensorPackObj operator*(const Transpose<SO3vec>& _y) const{
      const SO3vec& y=_y.obj;
      CtensorPackObj R(tau,y.tau,nbu,fill::zero);
      add_inp_into(R,y);
      return R;
    }
    */


  public: // ---- I/O --------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3vec";
    }

    string describe() const{
      return "SO3vec"+tau.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      if(fmt==0){
	for(int l=0; l<parts.size(); l++){
	  if(!parts[l]) continue;
	  oss<<indent<<"Part l="<<l<<":\n";
	  oss<<parts[l]->str(indent+"  ");
	  oss<<endl;
	}
      }
      if(fmt==1){
	//cout<<"Joint format"<<endl;
	for(int l=0; l<tau.size(); l++){
	  //COUT("printin"<<l);
	  oss<<indent<<"Part l="<<l<<":\n";
	  if(tau[l]>0) oss<<get_part(l).str(indent+"  ");
	  oss<<endl;
	}
      }
      return oss.str();
    }

    string repr(const string indent="") const{
      return "<GElib::SO3vec of type"+tau.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const SO3vec& x){
      stream<<x.str(); return stream;
    }
    
  };
  
  
  // ---- Post-class functions -------------------------------------------------------------------------------

  
  inline cnine::CscalarObj norm2(const SO3vec& x){
    cnine::CscalarObj r(x.nbu,cnine::fill::zero);
    x.norm2_into(r);
    return r;
  }

  inline cnine::CscalarObj inp(const SO3vec& x, const SO3vec& y){
    cnine::CscalarObj r(x.get_nbu(),cnine::fill::zero);
    x.add_inp_into(r,y);
    return r;
  }

  inline SO3vec operator*(const cnine::CscalarObj& c, const SO3vec& x){
    return x*c; 
  }

  inline SO3vec operator*(const cnine::CtensorPackObj& W, const SO3vec& x){
    return x.times(W);
  }

  inline SO3vec CGproduct(const SO3vec& x, const SO3vec& y, const int maxL=-1){
    SO3vec R(CGproduct(x.tau,y.tau,maxL),x.nbu,cnine::fill::zero,x.fmt,x.get_dev());
    R.add_CGproduct(x,y,maxL);
    return R;
  }


  //inline SO3vec SO3vecSeed::spawn(const fill_zero& fill){
  //return SO3vec(tau,nbu,fill::zero,device);
  //}


  /*
  inline SO3vec_part& SO3vec_part::operator=(const SO3part& x){
    assert(x.l==l);
    assert(x.nbu==nbu);
    //delete hdl;
    //hdl=Cengine_engine->push<ctensor_copy_op>(x.hdl);
    n=x.n;
    dims=x.dims;
    owner->tau[l]=n;
    return *this;
  }


  inline  SO3vec_part& SO3vec_part::operator=(SO3part&& x){
    assert(x.l==l);
    assert(x.nbu==nbu);
    delete hdl;
    //hdl=x.hdl;
    //x.hdl=nullptr;
    n=x.n;
    dims=x.dims;
    return *this;
  }
  */



  // ---- Free-standing functions ----------------------------------------------------------------------------


  

  // ---- Downcasting ----------------------------------------------------------------------------------------


  /*
  inline SO3vec& asSO3vec(Dnode* x){
    if(!dynamic_cast<SO3vec*>(x->obj)){
      if(!x->obj) cerr<<"GElib error: Dobject does not exist."<<endl;
      else {cerr<<"GElib error: Dobject is of type "<<x->obj->classname()<<" instead of SO3vec."<<endl;}
    }
    assert(dynamic_cast<SO3vec*>(x->obj));
    return *static_cast<SO3vec*>(x->obj);
  }

  inline const SO3vec& asSO3vec(const Dnode* x){
    if(!dynamic_cast<const SO3vec*>(x->obj)){
      if(!x->obj) cerr<<"GElib error: Dobject does not exist."<<endl;
      else {cerr<<"GElib error: Dobject is of type "<<x->obj->classname()<<" instead of SO3vec."<<endl;}
    }
    assert(dynamic_cast<const SO3vec*>(x->obj));
    return *static_cast<const SO3vec*>(x->obj);
  }

  inline SO3vec& asSO3vec(Dobject* x){
    if(!dynamic_cast<SO3vec*>(x)){
      if(!x) cerr<<"GElib error: Dobject does not exist."<<endl;
      else {cerr<<"GElib error: Dobject is of type "<<x->classname()<<" instead of SO3vec."<<endl;}
    }
    assert(dynamic_cast<SO3vec*>(x));
    return *static_cast<SO3vec*>(x);
  }

  inline const SO3vec& asSO3vec(const Dobject* x){
    if(!dynamic_cast<const SO3vec*>(x)){
      if(!x) cerr<<"GElib error: Dobject does not exist."<<endl;
      else {cerr<<"GElib error: Dobject is of type "<<x->classname()<<" instead of SO3vec."<<endl;}
    }
    assert(dynamic_cast<const SO3vec*>(x));
    return *static_cast<const SO3vec*>(x);
  }

  inline SO3vec& asSO3vec(Dobject& x){
    if(!dynamic_cast<SO3vec*>(&x))
      cerr<<"GElib error: Dobject is of type "<<x.classname()<<" instead of SO3vec."<<endl;
    assert(dynamic_cast<SO3vec*>(&x));
    return static_cast<SO3vec&>(x);
  }

  inline const SO3vec& asSO3vec(const Dobject& x){
    if(!dynamic_cast<const SO3vec*>(&x))
      cerr<<"GElib error: Dobject is of type "<<x.classname()<<" instead of SO3vec."<<endl;
    assert(dynamic_cast<const SO3vec*>(&x));
    return static_cast<const SO3vec&>(x);
  }
  */


}

#endif 

    //hdl=engine::new_ctensor({2*l+1,n},-1,device);
    //hdl=engine::new_ctensor({2*l+1,n},-1,device);
    //hdl=engine::new_ctensor_zero({2*l+1,n},-1,device);
    //hdl=engine::new_ctensor_gaussian({2*l+1,n},-1,device);
    //hdl(engine::ctensor_copy(x.hdl)){}
  /*
  class SO3vecSeed{
  public:
    SO3type tau; 
    int nbu=-1;
    int device=0; 
    SO3vecSeed(const SO3type& _tau, const int _nbu, const int _device=0): 
      tau(_tau), nbu(_nbu), device(_device){}
    ~SO3vecSeed(){}
    SO3vec spawn(const fill_zero& fill);  
  };
  */

    /*
    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const int _device=0): 
      tau(_tau), nbu(_nbu), device(_device){
      for(int l=0; l<tau.size(); l++)
	if(tau[l]>0) parts.push_back(new SO3part(l,tau[l],nbu,fill,_device));
	else parts.push_back(nullptr);
    }
    */

    //template<typename FILLTYPE, typename = typename 
    //     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SO3vec(const SO3type& _tau, const FILLTYPE fill, const dev_id& _dev, const format& _format): 
    //SO3vec(_tau,-1,fill,_dev.id(),toint(_format)){}

    //template<typename FILLTYPE, typename = typename 
    //std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SO3vec(const SO3type& _tau, const FILLTYPE fill, const device_id& _dev, const int _fmt): 
    //SO3vec(_tau,-1,fill,_device.id(),_fmt){}

    /*
    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const FILLTYPE fill, const format& _fmt, const device_id& _dev): 
      SO3vec(_tau,-1,fill,_fmt,_dev.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const format& _fmt, const device_id& _device): 
      SO3vec(_tau,nbu,fill,_fmt,_device.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const device_id& _device): 
      tau(_tau), nbu(_nbu), dev(_device.id()){
      for(int l=0; l<tau.size(); l++)
	if(tau[l]>0) parts.push_back(new SO3part(l,tau[l],nbu,fill,_device.id()));
	else parts.push_back(nullptr);
    }
    */

   /*
    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const format& _format, const int _dev=0): 
      tau(_tau), nbu(_nbu), dev(_dev){
      if(_format==format::basic){
	for(int l=0; l<tau.size(); l++)
	  if(tau[l]>0) parts.push_back(new SO3part(l,tau[l],nbu,fill,_dev));
	  else parts.push_back(nullptr);
      }
      if(_format==format::compact){
	fmt=1;
	if(eqTypes<FILLTYPE,fill_zero>()) hdl=Cengine_engine->push<new_SO3vec_zero_op>(tau,nbu,dev); 
	if(eqTypes<FILLTYPE,fill_ones>()) hdl=Cengine_engine->push<new_SO3vec_ones_op>(tau,nbu,dev);
	if(eqTypes<FILLTYPE,fill_gaussian>()) hdl=Cengine_engine->push<new_SO3vec_gaussian_op>(tau,nbu,dev);
      }
    }
    */

    /*
    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const FILLTYPE fill, const int _dev=0): 
      SO3vec(_tau,-1,fill,_dev){}
    */

    /*
    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const SO3type& _tau, const int _nbu, const FILLTYPE fill, const format& _format, const int _dev=0): 
      tau(_tau), nbu(_nbu), dev(_dev){
      if(_format==format::basic){
	for(int l=0; l<tau.size(); l++)
	  if(tau[l]>0) parts.push_back(new SO3part(l,tau[l],nbu,fill,_dev));
	  else parts.push_back(nullptr);
      }
      if(_format==format::compact){
	fmt=1;
	if(eqTypes<FILLTYPE,fill_zero>()) hdl=Cengine_engine->push<new_SO3vec_zero_op>(tau,nbu,dev); 
	if(eqTypes<FILLTYPE,fill_ones>()) hdl=Cengine_engine->push<new_SO3vec_ones_op>(tau,nbu,dev);
	if(eqTypes<FILLTYPE,fill_gaussian>()) hdl=Cengine_engine->push<new_SO3vec_gaussian_op>(tau,nbu,dev);
      }
    }
    */


  // ----- SO3vecElement -------------

  /*
  inline SO3vecElement::operator cscalar() const{
    return obj.get(l,i,m);
  }

  inline SO3vecElement& SO3vecElement::operator=(const cscalar& x){
    obj.set(l,i,m,x);    
    return *this;
  }

  inline complex<float> SO3vecElement::get_value() const{
    return obj.get_value(l,i,m);
  }
  
  inline SO3vecElement& SO3vecElement::set_value(const complex<float> x){
    obj.set_value(l,i,m,x);
    return *this;
  }

  inline constSO3vecElement::operator CscalarObj() const{
    return obj.get(l,i,m);
  }

  inline complex<float> constSO3vecElement::get_value() const{
    return obj.get_value(l,i,m);
  }

  ostream& operator<<(ostream& stream, const SO3vecElement& x){
    stream<<x.str(); return stream;
  }

  ostream& operator<<(ostream& stream, const constSO3vecElement& x){
    stream<<x.str(); return stream;
  }

  */
    //SO3vec& to_device(const device_id& _dev){
    //return to_device(_dev.id());
    //}

    /*
    SO3vec& to_device(const int _dev){
      dev=_dev;
      for(auto p:parts)
	if(p) p->to_device(_dev);
      //if(fmt==1) Cengine_engine->push<SO3vec_to_device_op>(_dev);
      return *this;
    }
    */
    /*
    const SO3vec& to_format(const SO3vec_format& _fmt) const{
      return to_format(toint(_fmt));
    }

    const SO3vec& to_format(const int _fmt) const{
      if(fmt==_fmt) return *this;
      SO3vec& me=const_cast<SO3vec&>(*this);

      if(fmt==0){
	if(_fmt==1){
	  SO3vec t(parts,_fmt,dev,true); // boooo
	  //for(auto p:parts) p=nullptr;
	  me.parts.clear();
	  me=std::move(t);
	}
	return *this;
      }

      if(fmt==1){
	GENET_ERROR("to_format not yet implemented.");
      }

      return *this;
    }
    */
    /*
    const SO3part& get(const int l) const{
      cout<<*parts[l]<<endl; 
      return *parts[l];
    }
    */

    /*
    const SO3part& operator[](const int l) const{
      assert(fmt==0);
      return *parts[l];
    }

    const SO3part& part(const int l) const{
      assert(fmt==0);
      return *parts[l];
    }

    SO3part& part(const int l){
      assert(fmt==1);
      return *parts[l];
    }
    */


    /*
    Dobject* clone() const{
      return new SO3vec(*this);
    }

    Dobject* spawn(const cnine::fill_zero& fill) const{
      return new SO3vec(tau,nbu,fill::zero,fmt,dev);
    }

    Dobject* spawn(const fill_zero& fill, const int dev) const{
      return new SO3vec(tau,nbu,fill::zero,fmt,dev);
    }

    Dobject* spawn(const SO3type& _tau, const fill_zero& fill) const{
      return new SO3vec(_tau,nbu,fill::zero,fmt,dev);
    }

    Dobject* spawn(const fill_gaussian& fill) const{
      return new SO3vec(tau,nbu,fill::gaussian,fmt,dev);
    }

    Dobject* spawn_part(const int l, const fill_zero& fill) const{
      assert(fmt==0);
      return new SO3part(l,tau[l],fill::zero,dev);
    }
    */

    // ---- get_value/set_value 

    /*
    complex<float> get_value(const int l, const int i, const int m) const{
      GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);
      assert(l<=parts.size());
      return parts[l]->get_value(i,m);
    }

    SO3vec& set_value(const int l, const int i, const int m, complex<float> x){
      GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);
      assert(l<=parts.size());
      assert(parts[l]);
      parts[l]->set(i,m,x);
      return *this; 
    }
    */


    // ---- get/set 

    /*
    cscalar operator()(const int l, const int i, const int m) const{
      GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);
      assert(l<=parts.size());
      assert(parts[l]);
      return parts[l]->get(i,m);
    }

    cscalar get(const int l, const int i, const int m) const{
      GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);
      assert(l<=parts.size());
      assert(parts[l]);
      return parts[l]->get(i,m);
    }

    SO3vec& set(const int l, const int i, const int m, const cscalar& x){
      GELIB_CHECK_SO3VEC_FORMAT_IS_0(fmt);
      assert(l<=parts.size());
      assert(parts[l]);
      parts[l]->set(i,m,x);
      return *this; 
    }
    */

    // ---- MemberExpr
    
    /*
    MemberExpr3<SO3vec,cscalar,complex<float>,int,int,int> operator()(const int l, const int i, const int m){
      return MemberExpr3<SO3vec,cscalar,complex<float>,int,int,int>(*this,l,i,m);
    }

    constMemberExpr3<SO3vec,cscalar,complex<float>,int,int,int> operator()(const int l, const int i, const int m) const{
      return constMemberExpr3<SO3vec,cscalar,complex<float>,int,int,int>(*this,l,i,m);
    }
    */


    /*
    SO3vec format(const SO3vec_format& _fmt) const{
      if(fmt==toint(_fmt)) return *this;
      return SO3vec(*this,_fmt);
    }

    SO3vec format(const int _fmt) const{
      if(fmt==_fmt) return *this;
      return SO3vec(*this,_format(_fmt));
    }
    */

    //const SO3vec& to_format(const int _fmt) const{
    //if(fmt==_fmt) return *this;
    //SO3vec& me=const_cast<SO3vec&>(*this);
    //}
      /*
    void add_sum(const vector<SO3vec*> v){
      const int N=v.size();
      for(auto p:v){
	add(*p);
      }
      return;
      
      if(fmt==0){
	vector<SO3part*> sub(N);
	for(int l=0; l<parts.size(); l++){
	  if(!parts[l]) continue; 
	  for(int i=0; i<N; i++)
	    sub[i]=v[i]->parts[l];
	  parts[l]->add_sum(sub);
	}
      }

      if(fmt==1){
	vector<Chandle*> h(N);
	for(int i=0; i<N; i++){
	  assert(v[i]->fmt==1);
	  h[i]=v[i]->hdl;
	}
	replace(hdl,Cengine_engine->push<SO3vec_add_sum_op>(hdl,h));
      }

    }
      */
