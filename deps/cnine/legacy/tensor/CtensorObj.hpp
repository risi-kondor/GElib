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


#ifndef _CnineCtensorObj
#define _CnineCtensorObj

#include "Cnine_base.hpp"
#include "CtensorA.hpp"
#include "CtensorB.hpp"
//#include "Dobject.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "RtensorObj.hpp"
#include "CtensorObj_helpers.hpp"
//#include "GenericOperators.hpp"


namespace cnine{


  class CtensorObj: public CNINE_CTENSOR_IMPL{
  public:

    //using CNINE_CTENSOR_IMPL::CNINE_CTENSOR_IMPL; 

    static complex<float> dummy_scalar() {return 0;}

#ifdef WITH_FAKE_GRAD
    CtensorObj* grad=nullptr;
#endif 

    ~CtensorObj(){
#ifdef WITH_FAKE_GRAD
      if(!is_view) delete grad;
#endif 
    }


  public: // ---- Public Constructors ------------------------------------------------------------------------


    CtensorObj(const Gdims& _dims, const int _dev=0):
      CNINE_CTENSOR_IMPL(_dims,fill_raw(),_dev){}

    CtensorObj(const Gdims& _dims, const int _nbu, const int _dev):
      CNINE_CTENSOR_IMPL(_dims,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorObj(const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      CNINE_CTENSOR_IMPL(_dims,fill,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorObj(const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      CNINE_CTENSOR_IMPL(_dims,fill,_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    //static CtensorObj raw(const Gdims& _dims, const int nbd=-1, const int _dev=0){
    //return CtensorObj(_dims,nbd,fill::raw,_dev);}
    //static CtensorObj raw(const Gdims& _dims, const int nbd, const device& _dev){
    //return CtensorObj(_dims,nbd,fill::raw,_dev.id());}
    static CtensorObj raw(const Gdims& _dims, const int _dev=0){
      return CtensorObj(_dims,fill::raw,_dev);}
    static CtensorObj raw(const Gdims& _dims, const struct device& _dev){
      return CtensorObj(_dims,fill::raw,_dev.id());}

    //static CtensorObj zero(const Gdims& _dims, const int nbd=-1, const int _dev=0){
    //return CtensorObj(_dims,nbd,fill::zero,_dev);}
    //static CtensorObj zero(const Gdims& _dims, const int nbd, const struct device& _dev){
    //return CtensorObj(_dims,nbd,fill::zero,_dev.id());}
    static CtensorObj zero(const Gdims& _dims, const int _dev=0){
      return CtensorObj(_dims,fill::zero,_dev);}
    static CtensorObj zero(const Gdims& _dims, const struct device& _dev){
      return CtensorObj(_dims,-1,fill::zero,_dev.id());}

    static CtensorObj zeros(const Gdims& _dims, const int _dev=0){
      return CtensorObj(_dims,fill::zero,_dev);}
    static CtensorObj zeros(const Gdims& _dims, const struct device& _dev){
      return CtensorObj(_dims,-1,fill::zero,_dev.id());}

    //static CtensorObj ones(const Gdims& _dims, const int nbd=-1, const int _dev=0){
    //return CtensorObj(_dims,nbd,fill::ones,_dev);}
    //static CtensorObj ones(const Gdims& _dims, const int nbd, const struct device& _dev){
    //return CtensorObj(_dims,nbd,fill::ones,_dev.id());}
    static CtensorObj ones(const Gdims& _dims, const int _dev=0){
      return CtensorObj(_dims,fill::ones,_dev);}
    static CtensorObj ones(const Gdims& _dims, const struct device& _dev){
      return CtensorObj(_dims,fill::ones,_dev.id());}

    //static CtensorObj identity(const Gdims& _dims, const int nbd=-1, const int _dev=0){
    //return CtensorObj(_dims,nbd,fill::identity,_dev);}
    //static CtensorObj identity(const Gdims& _dims, const int nbd, const struct device& _dev){
    //return CtensorObj(_dims,nbd,fill::identity,_dev.id());}
    static CtensorObj identity(const Gdims& _dims, const int _dev=0){
      return CtensorObj(_dims,fill::identity,_dev);}
    static CtensorObj identity(const Gdims& _dims, const struct device& _dev){
      return CtensorObj(_dims,fill::identity,_dev.id());}

    //static CtensorObj sequential(const Gdims& _dims, const int nbd=-1, const int _dev=0){
    //return CtensorObj(_dims,nbd,fill::sequential,_dev);}
    //static CtensorObj sequential(const Gdims& _dims, const int nbd, const struct device& _dev){
    //return CtensorObj(_dims,nbd,fill::sequential,_dev.id());}
    static CtensorObj sequential(const Gdims& _dims, const int _dev=0){
      return CtensorObj(_dims,fill::sequential,_dev);}
    static CtensorObj sequential(const Gdims& _dims, const struct device& _dev){
      return CtensorObj(_dims,fill::sequential,_dev.id());}

    //static CtensorObj gaussian(const Gdims& _dims, const int nbd=-1, const int _dev=0){
    //return CtensorObj(_dims,nbd,fill::gaussian,_dev);}
    //static CtensorObj gaussian(const Gdims& _dims, const int nbd, const struct device& _dev){
    //return CtensorObj(_dims,nbd,fill::gaussian,_dev.id());}
    static CtensorObj gaussian(const Gdims& _dims, const int _dev=0){
      return CtensorObj(_dims,fill::gaussian,_dev);}
    static CtensorObj gaussian(const Gdims& _dims, const struct device& _dev){
      return CtensorObj(_dims,fill::gaussian,_dev.id());}


    static CtensorObj zeros_like(const CtensorObj& x){
      return CtensorObj::zeros(x.get_dims(),x.get_dev());
    }


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static CtensorObj* new_zeros_like(const CtensorObj& x){
      return new CtensorObj(x.get_dims(),fill_zero(),x.get_dev());
    }

    
  public: // ---- Lambda constructors ------------------------------------------------------------------------


    CtensorObj(const Gdims& _dims, std::function<complex<float>(const int i, const int j)> fn):
      CNINE_CTENSOR_IMPL(_dims,fill_raw()){
      assert(get_ndims()==2);
      for(int i=0; i<get_dim(0); i++)
	for(int j=0; j<get_dim(1); j++)
	  set(i,j,fn(i,j));
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    CtensorObj(const CNINE_CTENSOR_IMPL& x):
      CNINE_CTENSOR_IMPL(x){};
      
    CtensorObj(CNINE_CTENSOR_IMPL&& x):
      CNINE_CTENSOR_IMPL(std::move(x)){};
      
    CtensorObj(const CtensorObj& x):
      CNINE_CTENSOR_IMPL(x){
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new CtensorObj(x);
      #endif
    };
      
    CtensorObj(const CtensorObj& x, const int _dev):
      CNINE_CTENSOR_IMPL(x,_dev){
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new CtensorObj(x);
      #endif
    };
      
    CtensorObj(const CtensorObj& x, const struct device& _dev):
      CNINE_CTENSOR_IMPL(x,_dev.id()){
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new CtensorObj(x);
      #endif
    };
      
    CtensorObj(const CtensorObj& x, const fill_zero& dummy):
      CtensorObj(x.dims,x.dev){}
      
    CtensorObj(CtensorObj&& x):
      CNINE_CTENSOR_IMPL(std::move(x)){
      #ifdef WITH_FAKE_GRAD
      grad=x.grad;
      x.grad=nullptr;
      #endif
    };

    CtensorObj& operator=(const CtensorObj& x){
      CNINE_CTENSOR_IMPL::operator=(x);
      #ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      if(x.grad) grad=new CtensorObj(x);
      #endif
      return *this;
    }

    CtensorObj& operator=(CtensorObj&& x){
      CNINE_CTENSOR_IMPL::operator=(std::move(x));
      #ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      grad=x.grad;
      x.grad=nullptr;
      #endif
      return *this;
    }
    
    //template<typename FILLTYPE>
    //CnineObject* spawn(const FILLTYPE& fill) const{
    //return new CtensorObj(dims,nbu,fill,dev);
    //}

    //template<typename FILLTYPE>
    //CnineObject* spawn(const FILLTYPE& fill, const int _dev) const{
    //return new CtensorObj(dims,nbu,fill,_dev);
    //}

    //CnineObject* spawn_zero() const{
    //return new CtensorObj(dims,nbu,fill::zero,dev);
    //}


    /*
    Dobject* clone() const{
      return new CtensorObj(*this);
    }

    Dobject* spawn(const fill_zero& fill) const{
      return new CtensorObj(CNINE_CTENSOR_IMPL::spawn(fill));
    }

    Dobject* spawn(const fill_zero& fill, const int _dev) const{
      return new CtensorObj(CNINE_CTENSOR_IMPL::spawn(fill),_dev);
    }

    Dobject* spawn(const fill_gaussian& fill) const{
      return new CtensorObj(CNINE_CTENSOR_IMPL::spawn(fill));
    }
    */

    //CtensorObj(CtensorObj& x, const view_flag& flag):
    //CtensorObj(CNINE_CTENSOR_IMPL(x,flag)){}
      

  public: // ---- Views --------------------------------------------------------------------------------------


    CtensorObj view(){
      return CNINE_CTENSOR_IMPL::view();
    }


    CtensorObj* viewp(){
      return new CtensorObj(CNINE_CTENSOR_IMPL::view());
    }


  public: // ---- Conversions --------------------------------------------------------------------------------

    
    CtensorObj(const RtensorObj& re, const RtensorObj& im):
      CNINE_CTENSOR_IMPL(re,im){}

    RtensorObj real() const{
      return RtensorObj(CNINE_CTENSOR_IMPL::real());
    }

    RtensorObj imag() const{
      return RtensorObj(CNINE_CTENSOR_IMPL::imag());
    }


    CtensorObj(const Conjugate<CtensorObj>& x):
      CtensorObj(x.obj.conj()){}

    CtensorObj(const Transpose<CtensorObj>& x):
      CtensorObj(x.obj.transp()){}

    CtensorObj(const Hermitian<CtensorObj>& x):
      CtensorObj(x.obj.herm()){}

    CtensorObj(const Transpose<Conjugate<CtensorObj> >& x):
      CtensorObj(x.obj.obj.herm()){}

    CtensorObj(const Conjugate<Transpose<CtensorObj> >& x):
      CtensorObj(x.obj.obj.herm()){}

    CtensorObj(const Gtensor<complex<float> >& x, const struct device& _dev):
      CNINE_CTENSOR_IMPL(x,_dev.id()){}


  public: // ---- ATEN ---------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    static CtensorObj view(at::Tensor& T){
      return CNINE_CTENSOR_IMPL::view(T);
    }

    static CtensorObj* viewp(at::Tensor& T){
      return new CtensorObj(CNINE_CTENSOR_IMPL::view(T));
    }

    CtensorObj(const at::Tensor& T):
      CNINE_CTENSOR_IMPL(T){}

    CtensorObj(const int dummy, const at::Tensor& T): // deprecated
      CNINE_CTENSOR_IMPL(dummy,T){}

#endif

  public: // ---- Transport ----------------------------------------------------------------------------------
  

    CtensorObj to_device(const int _dev) const{
      return CtensorObj(CNINE_CTENSOR_IMPL::to_device(_dev));
    }
  
  
 
  public: // ---- Access -------------------------------------------------------------------------------------


    //int get_nbu() const{ 
    //return nbu;
    //}

    int get_k() const{ 
      return dims.size();
    }

    int get_ndims() const{ 
      return dims.size();
    }

    Gdims get_dims() const{ 
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }

    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
    }


  public: // ---- Experimental -------------------------------------------------------------------------------


    #ifdef WITH_FAKE_GRAD
    void add_to_grad(const CtensorObj& x){
      if(grad) grad->add(x);
      else grad=new CtensorObj(x);
    }

    CtensorObj& get_grad(){
      if(!grad) grad=CtensorObj::new_zeros_like(*this);
      return *grad;
    }

    CtensorObj view_of_grad(){
      if(!grad) grad=new_zeros_like(*this);
      return grad->view();
    }
    #endif 


  public: // ---- Get/set elements ---------------------------------------------------------------------------
 

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

    CtensorObj& set(const Gindex& ix, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(ix,v.val);
      return *this;
    }
    
    CtensorObj& set(const int i0, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,v.val);
      return *this;
    }
    
    CtensorObj& set(const int i0, const int i1, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,i1,v.val);
      return *this;
    }
    
    CtensorObj& set(const int i0, const int i1, const int i2, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,i1,i2,v.val);
      return *this;
    }


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

    CtensorObj& set_value(const Gindex& ix, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(ix,v);
      return *this;
    }
    
    CtensorObj& set_value(const int i0, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,v);
      return *this;
    }
    
    CtensorObj& set_value(const int i0, const int i1, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,i1,v);
      return *this;
    }
    
    CtensorObj& set_value(const int i0, const int i1, const int i2, complex<float> v){
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

    CtensorObj& set(const Gindex& ix, complex<float> v){
      return set_value(ix,v);}
    CtensorObj& set(const int i0, complex<float> v){
      return set_value(i0,v);}
    CtensorObj& set(const int i0, const int i1, complex<float> v){
      return set_value(i0,i1,v);}
    CtensorObj& set(const int i0, const int i1, const int i2, complex<float> v){
      return set_value(i0,i1,i2,v);}


  public: // ---- CtensorElement -----------------------------------------------------------------------------
    

    CtensorObj_element operator()(const Gindex& ix){
      return CtensorObj_element(*this, ix);
    }

    CtensorObj_element operator()(const int i0){
      return CtensorObj_element(*this, Gindex(i0));
    }

    CtensorObj_element operator()(const int i0, const int i1){
      return CtensorObj_element(*this, Gindex(i0,i1));
    }

    CtensorObj_element operator()(const int i0, const int i1, const int i2){
      return CtensorObj_element(*this, Gindex(i0,i1,i2));
    }

    ConstCtensorObj_element operator()(const Gindex& ix) const{
      return ConstCtensorObj_element(*this, ix);
    }

    ConstCtensorObj_element operator()(const int i0) const{
      return ConstCtensorObj_element(*this, Gindex(i0));
    }

    ConstCtensorObj_element operator()(const int i0, const int i1) const{
      return ConstCtensorObj_element(*this, Gindex(i0,i1));
    }

    ConstCtensorObj_element operator()(const int i0, const int i1, const int i2) const{
      return ConstCtensorObj_element(*this, Gindex(i0,i1,i2));
    }


  public: // -------------------------------------------------------------------------------------------------

    /*
    CtensorObj& add_to_element(const Gindex& ix, const CscalarObj& v){
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
      set_zero();
    }


  public: // ---- Not in-place operations --------------------------------------------------------------------


    CtensorObj conj() const{
      return CtensorObj(CNINE_CTENSOR_IMPL::conj());
    }

    CtensorObj transp() const{
      return CtensorObj(CNINE_CTENSOR_IMPL::transp());
    }

    CtensorObj herm() const{
      return CtensorObj(CNINE_CTENSOR_IMPL::herm());
    }

    CtensorObj plus(const CtensorObj& x) const{
      return CtensorObj(CNINE_CTENSOR_IMPL::plus(x));
    }

    //CtensorObj apply(std::function<complex<float>(const complex<float>)> fn) const{
    //CtensorObj R(get_dims(),fill_raw());
    //return R;
    //}

    //CtensorObj apply(std::function<complex<float>(const int i, const int j, const complex<float>)> fn) const{
    //return CNINE_CTENSOR_IMPL(*this,fn);
    //}


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_mprod(const CtensorObj& x, const CtensorObj& y){
      add_Mprod_AA<0>(x,y);
    }

    void add_mprod_AT(const CtensorObj& x, const CtensorObj& y){
      add_Mprod_AT<0>(x,y);
    }

    void add_mprod_TA(const CtensorObj& x, const CtensorObj& y){
      add_Mprod_TA<0>(x,y);
    }

    void add_mprod_AC(const CtensorObj& x, const CtensorObj& y){
      add_Mprod_AA<2>(x,y);
    }

    void add_mprod_TC(const CtensorObj& x, const CtensorObj& y){
      add_Mprod_TA<2>(x,y);
    }

    void add_mprod_AH(const CtensorObj& x, const CtensorObj& y){
      add_Mprod_AT<2>(x,y);
    }

    void add_mprod_HA(const CtensorObj& x, const CtensorObj& y){
      add_Mprod_TA<1>(x,y);
    }



  public: // ---- Slices and chunks --------------------------------------------------------------------------


    CtensorObj chunk(const int ix, const int offs, const int n){
      return CNINE_CTENSOR_IMPL::chunk(ix,offs,n);
    }

    CtensorObj slice(const int ix, const int offs){
      return CNINE_CTENSOR_IMPL::slice(ix,offs);
    }

    /*
    void add_to_slice(const int ix, const int offs, const CtensorObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_to_chunk(const int ix, const int offs, const CtensorObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_chunk_op>(hdl,x.hdl,ix,offs));
    }
    */

    void add_to_slices(const int ix, vector<const CtensorObj*> _v){
      vector<const CNINE_CTENSOR_IMPL*> v(_v.size());
      for(int i=0; i<_v.size(); i++) v[i]=_v[i];
      CNINE_CTENSOR_IMPL::add_to_slices(ix,v);
    }
    
    template<typename... Args>
    void add_to_slices(const int ix, const CtensorObj& x0, Args... args){
      add_to_slices(ix,const_variadic_unroller(x0,args...));
    }

    /*
    void add_slice(const CtensorObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_add_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void set_chunk(const CtensorObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_set_chunk_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_chunk(const CtensorObj& x, const int ix, const int offs, const int n){
      replace(hdl,Cengine_engine->push<ctensor_add_chunk_op>(hdl,x.hdl,ix,offs,n));
    }
    
    CtensorObj slice(const int ix, const int offs) const{
      CtensorObj R(dims.remove(ix),fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_slice_op>(R.hdl,hdl,ix,offs));
      return R;
    }

    CtensorObj chunk(const int ix, const int offs, const int n=1) const{
      Gdims Rdims=dims;
      Rdims[ix]=n;
      CtensorObj R(Rdims,fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_chunk_op>(R.hdl,hdl,ix,offs,n));
      return R;
    }
    */


  public: // ---- Into operations ----------------------------------------------------------------------------


    //void inp_into(CscalarObj& R, const CtensorObj& y) const{
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,y.hdl,dims));
    //}

    //void norm2_into(CscalarObj& R) const{
    //R.val=
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,hdl,dims));
    //}

    void add_norm2_back(const CscalarObj& g, const CtensorObj& x){
      //add(x,g.val);
      //add_cconj(x,g.val);
    }


  // ---- In-place operators ---------------------------------------------------------------------------------


    CtensorObj& operator+=(const CtensorObj& y){
      add(y);
      return *this;
    }

    CtensorObj& operator-=(const CtensorObj& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


    CtensorObj operator+(const CtensorObj& y) const{
      CtensorObj R(*this);
      R.add(y);
      return R;
    }

    CtensorObj operator-(const CtensorObj& y) const{
      CtensorObj R(*this);
      R.subtract(y);
      return R;
    }

    //CtensorObj operator*(const CscalarObj& c) const{
    //CtensorObj R(get_dims(),get_nbu(),fill::zero);
    //R.add(*this,c);
    //return R;
    //}

    /*
    CtensorObj operator*(const CtensorObj& y) const{
      int I=dims.combined(0,dims.k()-1);
      int J=y.dims.combined(1,y.dims.k());
      CtensorObj R({I,J},fill::zero);
      R.add_mprod(*this,y);
      return R;
    }
    */

    CtensorObj operator*(const Transpose<CtensorObj>& y) const{
      int I=dims.combined(0,dims.size()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.size()-1);
      CtensorObj R(cnine::dims(I,J),fill::zero);
      R.add_mprod_AT(*this,y.obj);
      return R;
    }


  public: // ---- Normalization ------------------------------------------------------------------------------


    CtensorObj col_norms() const{
      Gdims _dims=get_dims();
      CtensorObj R(_dims.remove(_dims.size()-2),fill::zero,dev);
      //R.add_col_norms(*this);
      return R;
    }

    CtensorObj divide_cols(const CtensorObj& N) const{
      return *this; 
      //return CtensorObj(CNINE_CTENSOR_IMPL::divide_cols(N)); 
    }
    
    CtensorObj normalize_cols() const{
      return *this;
      //return CtensorObj(CNINE_CTENSOR_IMPL::normalize_cols()); 
    }
    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::CtensorObj";
    }

    string describe() const{
      return "Ctensor"+dims.str();
    } 

    string str(const string indent="") const{
      return CNINE_CTENSOR_IMPL::str(indent);
    }

    string repr() const{
      return "<cnine::ctensor"+dims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const CtensorObj& x){
      stream<<x.str(); return stream;}

    //friend ostream& operator<<(ostream& stream, CtensorObj x){
    //stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------

  /*
  inline CtensorObj CtensorSeed::spawn(const fill_zero& fill){
    //if(nch<0) return new SO3partB(l,n,fill::zero,dev);
    return CtensorObj(dims,nbu,fill::zero,dev);
  }
  */

  inline CtensorObj_element::operator CscalarObj() const{
    return obj.get(ix);
  }

  inline CtensorObj_element& CtensorObj_element::operator=(const CscalarObj& x){
    obj.set(ix,x);    
    return *this;
  }

  inline complex<float> CtensorObj_element::get_value() const{
    return obj.get_value(ix);
  }
  
  inline CtensorObj_element& CtensorObj_element::set_value(const complex<float> x){
    obj.set_value(ix,x);
    return *this;
  }


  inline ConstCtensorObj_element::operator CscalarObj() const{
    return obj.get(ix);
  }

  inline ConstCtensorObj_element::operator complex<float>() const{
    return obj.get_value(ix);
  }

  inline complex<float> ConstCtensorObj_element::value() const{
    return obj.get_value(ix);
  }


  inline CtensorObj& asCtensor(CnineObject* x){
    assert(x); 
    if(!dynamic_cast<CtensorObj*>(x))
      cerr<<"cnine error: object is of type "<<x->classname()<<" instead of CtensorObj."<<endl;
    assert(dynamic_cast<CtensorObj*>(x));
    return static_cast<CtensorObj&>(*x);
  }



  // ---------------------------------------------------------------------------------------------------------

  /*
  inline CtensorObj& asCtensor(Dobject* x){
    assert(x); 
    if(!dynamic_cast<CtensorObj*>(x))
      cerr<<"GEnet error: Dobject is of type "<<x->classname()<<" instead of CtensorObj."<<endl;
    assert(dynamic_cast<CtensorObj*>(x));
    return static_cast<CtensorObj&>(*x);
  }

  inline CtensorObj& asCtensor(Dobject& x){
    if(!dynamic_cast<CtensorObj*>(&x))
      cerr<<"GEnet error: Dobject is of type "<<x.classname()<<" instead of CtensorObj."<<endl;
    assert(dynamic_cast<CtensorObj*>(&x));
    return static_cast<CtensorObj&>(x);
  }
  */

  /*
  inline CtensorObj& asCtensor(Dnode* x){
    assert(x->obj); 
    if(!dynamic_cast<CtensorObj*>(x->obj))
      cerr<<"GEnet error: Dobject is of type "<<x->obj->classname()<<" instead of CtensorObj."<<endl;
    assert(dynamic_cast<CtensorObj*>(x->obj));
    return static_cast<CtensorObj&>(*x->obj);
  }

  inline CtensorObj& asCtensor(Dnode& x){
    if(!dynamic_cast<CtensorObj*>(x.obj))
      cerr<<"GEnet error: Dobject is of type "<<x.obj->classname()<<" instead of CtensorObj."<<endl;
    assert(dynamic_cast<CtensorObj*>(x.obj));
    return static_cast<CtensorObj&>(*x.obj);
  }
  */

}


#endif


      //return CscalarObj(Cengine_engine->direct<complex<float> >(hdl,[&i](Cobject& x){
      //  return CTENSORB(&x).get(i);
      //  }),dev);
    /*
    CtensorObj& set(const Gindex& ix, complex<float> v){
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
    //CtensorObj& set(const Gindex& ix, CscalarObj& v){
    //replace(hdl,Cengine_engine->push<ctensor_set_element_op>(hdl,v.hdl,ix));
    //return *this;
    //}
    
  //inline CtensorObj_element::operator complex<float>() const{
  //return obj.get_value(ix);
  //}


  /*
  class CtensorObj;

  class CtensorSeed{
  public:
    
    Gdims dims;
    int nbu=-1;
    int dev; 

    CtensorSeed(const Gdims& _dims, const int _nbu, const int _dev=0):
      dims(_dims), nbu(_nbu), dev(_dev){}

    CtensorObj spawn(const fill_zero& fill);

  };


  // ---------------------------------------------------------------------------------------------------------

  */
    /*
    CtensorObj(const fill_stack& dummy, int ix, const vector<const CtensorObj*> v){
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

    CtensorObj(const fill_cat& dummy, int ix, const vector<const CtensorObj*> v){
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
    CtensorObj(const fill_stack& dummy, const int ix,  const CtensorObj& x, Args... args):
      CtensorObj(fill::stack,ix,const_variadic_unroller(x,args...)){}

    template<typename... Args>
    CtensorObj(const int ix, const CtensorObj& x, Args... args):
      CtensorObj(fill::stack,ix,const_variadic_unroller(x,args...)){}

    template<typename... Args>
    CtensorObj(const fill_cat& dummy, const int ix,  const CtensorObj& x, Args... args):
      CtensorObj(dummy,ix,const_variadic_unroller(x,args...)){}
    */

    /*
    CtensorObj(const Gdims& _dims, 
      std::function<complex<float>(const int i, const int j)> fn, const int _dev=0): 
      dims(_dims), nbu(-1), dev(_dev){
      hdl=Cengine_engine->push<new_ctensor_fn2_op>(_dims,-1,fn,_dev);
    }

    CtensorObj(const Gdims& _dims, const int _nbu,  
      std::function<complex<float>(const int i, const int j)> fn, const int _dev=0): 
      dims(_dims), nbu(_nbu), dev(_dev){
      hdl=Cengine_engine->push<new_ctensor_fn2_op>(_dims,nbu,fn,_dev);
    }
    */

//public: // ---- Gtensor ------------------------------------------------------------------------------------

    /*
    Gtensor<complex<float> > gtensor() const{
      if(dev==0) return ::Cengine::ctensor_get(hdl);
      CtensorObj R(*this,device(0));
      return ::Cengine::ctensor_get(R.hdl);
    }
    */

