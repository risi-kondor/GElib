/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineCtensorPackObj
#define _CnineCtensorPackObj

#include "Cnine_base.hpp"
#include "CtensorA.hpp"
//#include "Dobject.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
#include "GdimsPack.hpp"
//#include "CtensorPackObj_helpers.hpp"
//#include "GenericOperators.hpp"


namespace cnine{


  class CtensorPackObj: public CtensorObj{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;

    typedef CNINE_CTENSOR_IMPL ctensori;

    /*
    class offset: public std::vector<int>{
    public:

      offset(){}

      offset(const GdimsPack& _dimsp): std::vector<int>(_dimsp.size()){
	int t=0; 
	for(int i=0; i<_dimsp.size(); i++){
	  (*this)[i]=t; 
	  t+=tau[i];
	}
      }

    };
    */


    //using CtensorObj::CtensorObj; 

    //using CtensorObj::dev; 

    //SO3type tau; 
    int nbu; 
    int dev=0; // is this redundant? 
    //GdimsPack dimsp;
    vector<CtensorObj*> tensors;
    //ctensori* vec=nullptr;
    
    int fmt=0;

    CtensorPackObj(){}

    ~CtensorPackObj(){
      for(auto p: tensors) delete p;  
      //delete vec;
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    //CtensorPackObj(const cnine::fill_noalloc& dummy, const SO3type& _tau, const int _nbu, const int _fmt, const int _dev):
    //tau(_tau), nbu(_nbu), fmt(_fmt), dev(_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const int _nbu, const FILLTYPE fill, const int _fmt, const int _dev): 
      nbu(_nbu), dev(_dev), fmt(_fmt){ //, dimsp(_dims){
      const int n=_dims.size(); 
      if(fmt==0){
	for(int i=0; i<n; i++)
	  tensors.push_back(new CtensorObj(_dims[i],fill,_dev));
      }
      if(fmt==1){
	CNINE_UNIMPL();
      }
    }


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const int n, const Gdims& _dims, const int _nbu, const FILLTYPE fill, const int _fmt=0, const int _dev=0): 
      nbu(_nbu), dev(_dev), fmt(_fmt){
      if(fmt==0){
	for(int i=0; i<n; i++){
	  tensors.push_back(new CtensorObj(_dims,fill,_dev));
	  //dimsp.push_back(_dims);
	}
      }
      if(fmt==1){
	CNINE_UNIMPL();
      }
    }
    

    // ---- without nbu

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill): 
      CtensorPackObj(_dims,-1,fill,0,0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill, const pack_format& _format): 
      CtensorPackObj(_dims,-1,fill,toint(_format),0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill, const device& _device): 
      CtensorPackObj(_dims,-1,fill,0,_device.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill, const pack_format& _format, const device& _device): 
      CtensorPackObj(_dims,-1,fill,toint(_format),_device.id()){}


    // ---- with nbu

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const int _nbu, const FILLTYPE fill): 
      CtensorPackObj(_dims,_nbu,fill,0,0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const int _nbu, const FILLTYPE fill, const pack_format& _format): 
      CtensorPackObj(_dims,_nbu,fill,toint(_format),0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const int _nbu, const FILLTYPE fill, const device& _device): 
      CtensorPackObj(_dims,_nbu,fill,0,_device.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const int _nbu, const FILLTYPE fill, const pack_format& _format, 
      const device& _device): 
      CtensorPackObj(_dims,_nbu,fill,toint(_format),_device.id()){}


    static CtensorPackObj zero(const int _n, const Gdims& _dims){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::zero);}
    static CtensorPackObj zero(const int _n, const Gdims& _dims, const pack_format& _format){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::zero,toint(_format));}
    static CtensorPackObj zero(const int _n, const Gdims& _dims, const device& _device){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::zero,_device.id());}
    static CtensorPackObj zero(const int _n, const Gdims& _dims, const pack_format& _format, const device& _device){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::zero,toint(_format),_device.id());}
    
    static CtensorPackObj zero(const GdimsPack& _dims){
      return CtensorPackObj(_dims,-1,cnine::fill::zero);}
    static CtensorPackObj zero(const GdimsPack& _dims, const pack_format& _format){
      return CtensorPackObj(_dims,-1,cnine::fill::zero,toint(_format));}
    static CtensorPackObj zero(const GdimsPack& _dims, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::zero,_device.id());}
    static CtensorPackObj zero(const GdimsPack& _dims, const pack_format& _format, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::zero,toint(_format),_device.id());}
    
    static CtensorPackObj ones(const GdimsPack& _dims){
      return CtensorPackObj(_dims,-1,cnine::fill::ones);}
    static CtensorPackObj ones(const GdimsPack& _dims, const pack_format& _format){
      return CtensorPackObj(_dims,-1,cnine::fill::ones,_format);}
    static CtensorPackObj ones(const GdimsPack& _dims, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::ones,_device);}
    static CtensorPackObj ones(const GdimsPack& _dims, const pack_format& _format, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::ones,_format,_device);}
    
    static CtensorPackObj sequential(const int _n, const Gdims& _dims){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::sequential);}
    static CtensorPackObj sequential(const int _n, const Gdims& _dims, const pack_format& _format){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::sequential,toint(_format));}
    static CtensorPackObj sequential(const int _n, const Gdims& _dims, const device& _device){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::sequential,0,_device.id());}
    static CtensorPackObj sequential(const int _n, const Gdims& _dims, const pack_format& _format, const device& _device){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::sequential,toint(_format),_device.id());}
    
    static CtensorPackObj sequential(const GdimsPack& _dims){
      return CtensorPackObj(_dims,-1,cnine::fill::sequential);}
    static CtensorPackObj sequential(const GdimsPack& _dims, const pack_format& _format){
      return CtensorPackObj(_dims,-1,cnine::fill::sequential,_format);}
    static CtensorPackObj sequential(const GdimsPack& _dims, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::sequential,_device);}
    static CtensorPackObj sequential(const GdimsPack& _dims, const pack_format& _format, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::sequential,_format,_device);}
    
    static CtensorPackObj gaussian(const int _n, const Gdims& _dims){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::gaussian);}
    static CtensorPackObj gaussian(const int _n, const Gdims& _dims, const pack_format& _format){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::gaussian,toint(_format));}
    static CtensorPackObj gaussian(const int _n, const Gdims& _dims, const device& _device){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::gaussian,_device.id());}
    static CtensorPackObj gaussian(const int _n, const Gdims& _dims, const pack_format& _format, const device& _device){
      return CtensorPackObj(_n,_dims,-1,cnine::fill::gaussian,toint(_format),_device.id());}
    
    static CtensorPackObj gaussian(const GdimsPack& _dims){
      return CtensorPackObj(_dims,-1,cnine::fill::gaussian);}
    static CtensorPackObj gaussian(const GdimsPack& _dims, const pack_format& _format){
      return CtensorPackObj(_dims,-1,cnine::fill::gaussian,toint(_format));}
    static CtensorPackObj gaussian(const GdimsPack& _dims, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::gaussian,_device.id());}
    static CtensorPackObj gaussian(const GdimsPack& _dims, const pack_format& _format, const device& _device){
      return CtensorPackObj(_dims,-1,cnine::fill::gaussian,toint(_format),_device.id());}
    
  
public: // ---- Copying ------------------------------------------------------------------------------------


    //CtensorPackObj(const CNINE_CTENSOR_IMPL& x):
    //CNINE_CTENSOR_IMPL(x){
    //};
  
    CtensorPackObj(const CtensorPackObj& x):
      CtensorObj(x), nbu(x.nbu), fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p));
	else tensors.push_back(nullptr);
    };
      
    CtensorPackObj(const CtensorPackObj& x, const int _dev):
      CtensorObj(x,_dev), nbu(x.nbu), fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p,_dev));
	else tensors.push_back(nullptr);
    };
      
    CtensorPackObj(const CtensorPackObj& x, const device& _dev):
      CtensorObj(x,_dev.id()), nbu(x.nbu), fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p,_dev.id()));
	else tensors.push_back(nullptr);
    };
      
    CtensorPackObj(const CtensorPackObj& x, const fill_zero& dummy):
      CtensorObj(x,dummy), nbu(x.nbu), fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p,dummy));
	else tensors.push_back(nullptr);
    };
      
    CtensorPackObj(CtensorPackObj&& x):
      CtensorObj(std::move(x)), nbu(x.nbu), fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(std::move(*p)));
	else tensors.push_back(nullptr);
    };

    CtensorPackObj& operator=(const CtensorPackObj& x){
      if(fmt==0) CtensorObj::operator=(x);
      else{
	for(auto p: tensors) delete p;
	tensors.clear(); 
	for(auto p: x.tensors)
	  if(p) tensors.push_back(new CtensorObj(*p));
	  else tensors.push_back(nullptr);
      }
      return *this;
    }

    CtensorPackObj& operator=(CtensorPackObj&& x){
      if(fmt==0) CtensorObj::operator=(std::move(x));
      else{
	for(auto p: tensors) delete p;
	tensors.clear(); 
	for(auto p: x.tensors)
	  tensors.push_back(p);
      }
      return *this;
    }
    
    /*
    Dobject* clone() const{
      return new CtensorPackObj(*this);
    }

    Dobject* spawn(const fill_zero& fill) const{
      return new CtensorPackObj(CNINE_CTENSOR_IMPL::spawn(fill));
    }

    Dobject* spawn(const fill_zero& fill, const int _dev) const{
      return new CtensorPackObj(CNINE_CTENSOR_IMPL::spawn(fill),_dev);
    }

    Dobject* spawn(const fill_gaussian& fill) const{
      return new CtensorPackObj(CNINE_CTENSOR_IMPL::spawn(fill));
    }
    */

    //CtensorPackObj(CtensorPackObj& x, const view_flag& flag):
    //CtensorPackObj(CNINE_CTENSOR_IMPL(x,flag)){}
      

  public: // ---- Conversions --------------------------------------------------------------------------------


    /*
    CtensorPackObj(const Conjugate<CtensorPackObj>& x):
      CtensorPackObj(x.obj.conj()){}

    CtensorPackObj(const Transpose<CtensorPackObj>& x):
      CtensorPackObj(x.obj.transp()){}

    CtensorPackObj(const Hermitian<CtensorPackObj>& x):
      CtensorPackObj(x.obj.herm()){}

    CtensorPackObj(const Transpose<Conjugate<CtensorPackObj> >& x):
      CtensorPackObj(x.obj.obj.herm()){}

    CtensorPackObj(const Conjugate<Transpose<CtensorPackObj> >& x):
      CtensorPackObj(x.obj.obj.herm()){}

    CtensorPackObj(const Gtensor<complex<float> >& x, const device& _dev=device(0)):
      CNINE_CTENSOR_IMPL(x,_dev.id()){}
    */


  public: // ---- Access -------------------------------------------------------------------------------------

    /*
    int get_nbu() const{ 
      if(bundle){
	assert(dims.size()>0); 
	return dims[0];
      }
      return -1;
    }
    */

    /*
    int get_k() const{ 
      return dims.size();
    }
    */

    /*
    GdimsPack get_dims() const{ 
      return dimsp;
    }
    */

    Gdims get_dims(const int i) const{
      assert(i<tensors.size());
      return tensors[i]->get_dims();
    }

    int get_dev() const{
      return dev;
    }


  public: // ---- Get/set elements ---------------------------------------------------------------------------
 


    /*
    CscalarObj get(const Gindex& ix) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(ix));
    }

    CscalarObj get(const int i0) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(i0));
    }

    CscalarObj get(const int i0, const int i1) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(i0,i1));
    }

    CscalarObj get(const int i0, const int i1, const int i2) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(i0,i1,i2));
    }

    CtensorPackObj& set(const Gindex& ix, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(ix,v);
      return *this;
    }
    
    CtensorPackObj& set(const int i0, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,v);
      return *this;
    }
    
    CtensorPackObj& set(const int i0, const int i1, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,i1,v);
      return *this;
    }
    
    CtensorPackObj& set(const int i0, const int i1, const int i2, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,i1,i2,v);
      return *this;
    }
    */


  public: // ---- Get/set value ------------------------------------------------------------------------------


    /*
    complex<float> get_value(const Gindex& ix) const{
      return CNINE_CTENSOR_IMPL::get_value(ix);
    }

    complex<float> get_value(const int i0) const{
      return CNINE_CTENSOR_IMPL::get_value(i0);
    }

    complex<float> get_value(const int i0, const int i1) const{
      return CNINE_CTENSOR_IMPL::get_value(i0,i1);
    }

    complex<float> get_value(const int i0, const int i1, const int i2) const{
      return CNINE_CTENSOR_IMPL::get_value(i0,i1,i2);
    }
    */

    /*
    CtensorPackObj& set_value(const Gindex& ix, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(ix,v);
      return *this;
    }
    
    CtensorPackObj& set_value(const int i0, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,v);
      return *this;
    }
    
    CtensorPackObj& set_value(const int i0, const int i1, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,i1,v);
      return *this;
    }
    
    CtensorPackObj& set_value(const int i0, const int i1, const int i2, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,i1,i2,v);
      return *this;
    }
    
    complex<float> value(const Gindex& ix) const{
      return get_value(ix);}
    complex<float> value(const int i0) const{
      return get_value(i0);}
    complex<float> value(const int i0, const int i1) const{
      return get_value(i0,i1);}
    complex<float> value(const int i0, const int i1, const int i2) const{
      return get_value(i0,i1,i2);}

    CtensorPackObj& set(const Gindex& ix, complex<float> v){
      return set_value(ix,v);}
    CtensorPackObj& set(const int i0, complex<float> v){
      return set_value(i0,v);}
    CtensorPackObj& set(const int i0, const int i1, complex<float> v){
      return set_value(i0,i1,v);}
    CtensorPackObj& set(const int i0, const int i1, const int i2, complex<float> v){
      return set_value(i0,i1,i2,v);}
    */




  public: // -------------------------------------------------------------------------------------------------

    /*
    CtensorPackObj& add_to_element(const Gindex& ix, const CscalarObj& v){
      replace(hdl,Cengine_engine->push<ctensor_add_to_element_op>(hdl,v.hdl,ix));
      return *this;
    }
    
    void add_element_into(CscalarObj& r, const Gindex& ix){
      replace(r.hdl,Cengine_engine->push<ctensor_add_element_op>(r.hdl,hdl,ix));
    }
    */

    /*
    int combined(const int a, const int b) const{
      return asCtensorB(hdl->node->obj).combined(a,b);
    }
    */

    void flush() const{
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      //set_zero();
    }


  public: // ---- Not in-place operations --------------------------------------------------------------------

    /*
    CtensorPackObj conj() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::conj());
    }

    CtensorPackObj transp() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::transp());
    }

    CtensorPackObj herm() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::herm());
    }

    CtensorPackObj plus(const CtensorPackObj& x){
      return CtensorPackObj(CNINE_CTENSOR_IMPL::plus(x));
    }

    CtensorPackObj apply(std::function<complex<float>(const complex<float>)> fn) const{
      return CNINE_CTENSOR_IMPL(*this,fn);
    }

    CtensorPackObj apply(std::function<complex<float>(const int i, const int j, const complex<float>)> fn) const{
      return CNINE_CTENSOR_IMPL(*this,fn);
    }
    */


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const CtensorPackObj& x){
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->add(*x.tensors[l]);
      }
      if(fmt==1)
	CtensorObj::add(x);
    }

    void add(const CtensorPackObj& x, const rscalar& c){
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->add(*x.tensors[l],c);
      }
      if(fmt==1)
	CtensorObj::add(x,c);
    }

    void add(const CtensorPackObj& x, const cscalar& c){
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->add(*x.tensors[l],c);
      }
      if(fmt==1)
	CtensorObj::add(x,c);
    }

    void subtract(const CtensorPackObj& x){
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->subtract(*x.tensors[l]);
      }
      if(fmt==1)
	CtensorObj::subtract(x);
    }



    /*
    void add_mprod(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_AA<0>(x,y);
    }

    void add_mprod_AT(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_AT<0>(x,y);
    }

    void add_mprod_TA(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_TA<0>(x,y);
    }

    void add_mprod_AC(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_AA<2>(x,y);
    }

    void add_mprod_TC(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_TA<2>(x,y);
    }

    void add_mprod_AH(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_AT<2>(x,y);
    }

    void add_mprod_HA(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_TA<1>(x,y);
    }
    */


  public: // ---- Slices and chunks --------------------------------------------------------------------------

    /*
    CtensorPackObj chunk(const int ix, const int offs, const int n){
      return CNINE_CTENSOR_IMPL::chunk(ix,offs,n);
    }

    CtensorPackObj slice(const int ix, const int offs){
      return CNINE_CTENSOR_IMPL::slice(ix,offs);
    }
    */

    /*
    void add_to_slice(const int ix, const int offs, const CtensorPackObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_to_chunk(const int ix, const int offs, const CtensorPackObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_chunk_op>(hdl,x.hdl,ix,offs));
    }
    */

    /*
    void add_to_slices(const int ix, vector<const CtensorPackObj*> _v){
      vector<const CNINE_CTENSOR_IMPL*> v(_v.size());
      for(int i=0; i<_v.size(); i++) v[i]=_v[i];
      CNINE_CTENSOR_IMPL::add_to_slices(ix,v);
    }
    
    template<typename... Args>
    void add_to_slices(const int ix, const CtensorPackObj& x0, Args... args){
      add_to_slices(ix,const_variadic_unroller(x0,args...));
    }
    */

    /*
    void add_slice(const CtensorPackObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_add_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void set_chunk(const CtensorPackObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_set_chunk_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_chunk(const CtensorPackObj& x, const int ix, const int offs, const int n){
      replace(hdl,Cengine_engine->push<ctensor_add_chunk_op>(hdl,x.hdl,ix,offs,n));
    }
    
    CtensorPackObj slice(const int ix, const int offs) const{
      CtensorPackObj R(dims.remove(ix),fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_slice_op>(R.hdl,hdl,ix,offs));
      return R;
    }

    CtensorPackObj chunk(const int ix, const int offs, const int n=1) const{
      Gdims Rdims=dims;
      Rdims[ix]=n;
      CtensorPackObj R(Rdims,fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_chunk_op>(R.hdl,hdl,ix,offs,n));
      return R;
    }
    */


  public: // ---- Into operations ----------------------------------------------------------------------------


    //void inp_into(CscalarObj& R, const CtensorPackObj& y) const{
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,y.hdl,dims));
    //}

    //void norm2_into(CscalarObj& R) const{
    //R.val=
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,hdl,dims));
    //}

  /*
    void add_norm2_back(const CscalarObj& g, const CtensorPackObj& x){
      add(x,g);
      add_cconj(x,g);
    }
  */

  // ---- In-place operators ---------------------------------------------------------------------------------


    CtensorPackObj& operator+=(const CtensorPackObj& y){
      add(y);
      return *this;
    }

    CtensorPackObj& operator-=(const CtensorPackObj& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


    CtensorPackObj operator+(const CtensorPackObj& y) const{
      CtensorPackObj R(*this);
      R.add(y);
      return R;
    }

    CtensorPackObj operator-(const CtensorPackObj& y) const{
      CtensorPackObj R(*this);
      R.subtract(y);
      return R;
    }

    CtensorPackObj operator*(const CscalarObj& c) const{
      CtensorPackObj R(*this,fill::zero);
      R.add(*this,c);
      return R;
    }

    /*
    CtensorPackObj operator*(const CtensorPackObj& y) const{
      int I=dims.combined(0,dims.k()-1);
      int J=y.dims.combined(1,y.dims.k());
      CtensorPackObj R({I,J},fill::zero);
      R.add_mprod(*this,y);
      return R;
    }
    */

  /*
    CtensorPackObj operator*(const Transpose<CtensorPackObj>& y) const{
      int I=dims.combined(0,dims.k()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.k()-1);
      CtensorPackObj R({I,J},fill::zero);
      R.add_mprod_AT(*this,y.obj);
      return R;
    }
  */

  public: // ---- Normalization ------------------------------------------------------------------------------

    /*
    CtensorPackObj col_norms() const{
      Gdims _dims=get_dims();
      CtensorPackObj R(_dims.remove(_dims.size()-2),get_nbu(),fill::zero,dev);
      R.add_col_norms(*this);
      return R;
    }

    CtensorPackObj divide_cols(const CtensorPackObj& N) const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::divide_cols(N)); 
    }
    
    CtensorPackObj normalize_cols() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::normalize_cols()); 
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Cnine::CtensorPackObj";
    }

    string describe() const{
      return "CtensorPack";
    } 

    string str(const string indent="") const{
      //if(fmt==1) return CNINE_CTENSOR_IMPL::str(indent);
      ostringstream oss;
      for(int l=0; l<tensors.size(); l++){
	if(!tensors[l]) continue;
	oss<<indent<<"tensor "<<l<<":\n";
	oss<<tensors[l]->str(indent+"  ");
	oss<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CtensorPackObj& x){
      stream<<x.str(); return stream;}

    //friend ostream& operator<<(ostream& stream, CtensorPackObj x){
    //stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------

  /*
  inline CtensorPackObj CtensorSeed::spawn(const fill_zero& fill){
    //if(nch<0) return new SO3partB(l,n,fill::zero,dev);
    return CtensorPackObj(dims,nbu,fill::zero,dev);
  }
  */



  // ---------------------------------------------------------------------------------------------------------

  /*
  inline CtensorPackObj& asCtensor(Dobject* x){
    assert(x); 
    if(!dynamic_cast<CtensorPackObj*>(x))
      cerr<<"GEnet error: Dobject is of type "<<x->classname()<<" instead of CtensorPackObj."<<endl;
    assert(dynamic_cast<CtensorPackObj*>(x));
    return static_cast<CtensorPackObj&>(*x);
  }

  inline CtensorPackObj& asCtensor(Dobject& x){
    if(!dynamic_cast<CtensorPackObj*>(&x))
      cerr<<"GEnet error: Dobject is of type "<<x.classname()<<" instead of CtensorPackObj."<<endl;
    assert(dynamic_cast<CtensorPackObj*>(&x));
    return static_cast<CtensorPackObj&>(x);
  }
  */

  /*
  inline CtensorPackObj& asCtensor(Dnode* x){
    assert(x->obj); 
    if(!dynamic_cast<CtensorPackObj*>(x->obj))
      cerr<<"GEnet error: Dobject is of type "<<x->obj->classname()<<" instead of CtensorPackObj."<<endl;
    assert(dynamic_cast<CtensorPackObj*>(x->obj));
    return static_cast<CtensorPackObj&>(*x->obj);
  }

  inline CtensorPackObj& asCtensor(Dnode& x){
    if(!dynamic_cast<CtensorPackObj*>(x.obj))
      cerr<<"GEnet error: Dobject is of type "<<x.obj->classname()<<" instead of CtensorPackObj."<<endl;
    assert(dynamic_cast<CtensorPackObj*>(x.obj));
    return static_cast<CtensorPackObj&>(*x.obj);
  }
  */

}


#endif


      //return CscalarObj(Cengine_engine->direct<complex<float> >(hdl,[&i](Cobject& x){
      //  return CTENSORB(&x).get(i);
      //  }),dev);
    /*
    CtensorPackObj& set(const Gindex& ix, complex<float> v){
      Cengine_engine->direct(hdl,[&ix,&v](Cobject& x){
	  asCtensorB(&x,__PRETTY_FUNCTION__).set(ix,v);
	});
      return *this;
    }
    */
    /*
    complex<float> operator()(const int i) const{
      return Cengine_engine->direct<complex<float> >(hdl,[&i](Cobject& x){
	  return CTENSORB(&x).get(i);
	});
    }

    complex<float> operator()(const int i0, const int i1) const{
      return Cengine_engine->direct<complex<float> >(hdl,[&i0,&i1](Cobject& x){
	  return CTENSORB(&x).get(i0,i1);
	});
    }

    complex<float> operator()(const int i0, const int i1, const int i2) const{
      return Cengine_engine->direct<complex<float> >(hdl,[&i0,&i1,&i2](Cobject& x){
	  return CTENSORB(&x).get(i0,i1,i2);
	});
    }
    */
    //CtensorPackObj& set(const Gindex& ix, CscalarObj& v){
    //replace(hdl,Cengine_engine->push<ctensor_set_element_op>(hdl,v.hdl,ix));
    //return *this;
    //}
    
  //inline CtensorPackObj_element::operator complex<float>() const{
  //return obj.get_value(ix);
  //}


  /*
  class CtensorPackObj;

  class CtensorSeed{
  public:
    
    Gdims dims;
    int nbu=-1;
    int dev; 

    CtensorSeed(const Gdims& _dims, const int _nbu, const int _dev=0):
      dims(_dims), nbu(_nbu), dev(_dev){}

    CtensorPackObj spawn(const fill_zero& fill);

  };


  // ---------------------------------------------------------------------------------------------------------

  */
    /*
    CtensorPackObj(const fill_stack& dummy, int ix, const vector<const CtensorPackObj*> v){
      assert(v.size()>0);
      const int N=v.size();
      const Gdims& dims0=v[0]->dims;
      assert(ix<=dims0.size());
      for(int i=1; i<N; i++) assert(v[i]->dims==dims0);
      for(int i=0; i<ix; i++) dims.push_back(dims0[i]);
      dims.push_back(N);
      for(int i=ix; i<dims0.size(); i++) dims.push_back(dims0[i]);
      hdl=Cengine_engine->push<new_ctensor_zero_op>(dims,-1,0);
      vector<const Chandle*> w(N);
      for(int i=0; i<N; i++) w[i]=v[i]->hdl;
      replace(hdl,Cengine_engine->push<ctensor_add_to_slices_op>(hdl,w,ix));
    }

    CtensorPackObj(const fill_cat& dummy, int ix, const vector<const CtensorPackObj*> v){
      assert(v.size()>0);
      const int N=v.size();
      const Gdims& dims0=v[0]->dims;
      assert(ix<dims0.size());
      dims=dims0;
      int t=0; 
      for(int i=0; i<N; i++) t+=v[i]->dims[ix];
      dims[ix]=t;
      hdl=Cengine_engine->push<new_ctensor_zero_op>(dims,-1,0);
      int offs=0;
      for(int i=0; i<N; i++){
	replace(hdl,Cengine_engine->push<ctensor_add_to_chunk_op>(hdl,v[i]->hdl,ix,offs));
	offs+=v[i]->dims[ix];
      }
    }

    template<typename... Args>
    CtensorPackObj(const fill_stack& dummy, const int ix,  const CtensorPackObj& x, Args... args):
      CtensorPackObj(fill::stack,ix,const_variadic_unroller(x,args...)){}

    template<typename... Args>
    CtensorPackObj(const int ix, const CtensorPackObj& x, Args... args):
      CtensorPackObj(fill::stack,ix,const_variadic_unroller(x,args...)){}

    template<typename... Args>
    CtensorPackObj(const fill_cat& dummy, const int ix,  const CtensorPackObj& x, Args... args):
      CtensorPackObj(dummy,ix,const_variadic_unroller(x,args...)){}
    */

    /*
    CtensorPackObj(const Gdims& _dims, 
      std::function<complex<float>(const int i, const int j)> fn, const int _dev=0): 
      dims(_dims), nbu(-1), dev(_dev){
      hdl=Cengine_engine->push<new_ctensor_fn2_op>(_dims,-1,fn,_dev);
    }

    CtensorPackObj(const Gdims& _dims, const int _nbu,  
      std::function<complex<float>(const int i, const int j)> fn, const int _dev=0): 
      dims(_dims), nbu(_nbu), dev(_dev){
      hdl=Cengine_engine->push<new_ctensor_fn2_op>(_dims,nbu,fn,_dev);
    }
    */

//public: // ---- Gtensor ------------------------------------------------------------------------------------

    /*
    Gtensor<complex<float> > gtensor() const{
      if(dev==0) return ::Cengine::ctensor_get(hdl);
      CtensorPackObj R(*this,device(0));
      return ::Cengine::ctensor_get(R.hdl);
    }
    */


/*
    // ---- Constructors --------------------------------------------------------------------------------------


    //CtensorPackObj(const cnine::fill_noalloc& dummy, const SO3type& _tau, const int _nbu, const int _fmt, const int _dev):
    //tau(_tau), nbu(_nbu), fmt(_fmt), dev(_dev){}


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const int _nbu, const FILLTYPE fill, const int _fmt, const int _dev): 
      nbu(_nbu), dev(_dev), fmt(_fmt){
      const int n=std::min(tau1.size(),tau2.size()); 
      if(fmt==0){
	for(int i=0; i<n; i++)
	  tensors.push_back(new tensori({tau2[i],tau1[i]},fill,_dev));
      }
      if(fmt==2){
	GELIB_UNIMPL();
      }
    }


    // ---- without nbu

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const FILLTYPE fill): 
      CtensorPackObj(_tau1,_tau2,-1,fill,0,0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const FILLTYPE fill, const pack_format& _format): 
      CtensorPackObj(_tau1,_tau2,-1,fill,toint(_format),0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const FILLTYPE fill, const device& _device): 
      CtensorPackObj(_tau1,_tau2,-1,fill,0,_device.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const FILLTYPE fill, const pack_format& _format, const device& _device): 
      CtensorPackObj(_tau1,_tau2,-1,fill,toint(_format),_device.id()){}


    // ---- with nbu

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const int _nbu, const FILLTYPE fill): 
      CtensorPackObj(_tau1,_tau2,_nbu,fill,0,0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const int _nbu, const FILLTYPE fill, const pack_format& _format): 
      CtensorPackObj(_tau1,_tau2,_nbu,fill,toint(_format),0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const int _nbu, const FILLTYPE fill, const device& _device): 
      CtensorPackObj(_tau1,_tau2,_nbu,fill,0,_device.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const SO3type& _tau1, const SO3type& _tau2, const int _nbu, const FILLTYPE fill, const pack_format& _format, 
      const device& _device): 
      CtensorPackObj(_tau1,_tau2,_nbu,fill,toint(_format),_device.id()){}


    static CtensorPackObj zero(const SO3type& _tau1, const SO3type& _tau2){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::zero);}
    static CtensorPackObj zero(const SO3type& _tau1, const SO3type& _tau2, const pack_format& _format){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::zero,_format);}
    static CtensorPackObj zero(const SO3type& _tau1, const SO3type& _tau2, const device& _device){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::zero,_device);}
    static CtensorPackObj zero(const SO3type& _tau1, const SO3type& _tau2, const pack_format& _format, const device& _device){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::zero,_format,_device);}
    
    static CtensorPackObj ones(const SO3type& _tau1, const SO3type& _tau2){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::ones);}
    static CtensorPackObj ones(const SO3type& _tau1, const SO3type& _tau2, const pack_format& _format){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::ones,_format);}
    static CtensorPackObj ones(const SO3type& _tau1, const SO3type& _tau2, const device& _device){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::ones,_device);}
    static CtensorPackObj ones(const SO3type& _tau1, const SO3type& _tau2, const pack_format& _format, const device& _device){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::ones,_format,_device);}
    
    static CtensorPackObj gaussian(const SO3type& _tau1, const SO3type& _tau2){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::gaussian);}
    static CtensorPackObj gaussian(const SO3type& _tau1, const SO3type& _tau2, const pack_format& _format){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::gaussian,_format);}
    static CtensorPackObj gaussian(const SO3type& _tau1, const SO3type& _tau2, const device& _device){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::gaussian,_device);}
    static CtensorPackObj gaussian(const SO3type& _tau1, const SO3type& _tau2, const pack_format& _format, const device& _device){
      return CtensorPackObj(_tau1,_tau2,-1,cnine::fill::gaussian,_format,_device);}
*/    

