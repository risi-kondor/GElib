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


#ifndef _CnineCtensorArray
#define _CnineCtensorArray

#include "Cnine_base.hpp"
#include "CtensorArrayA.hpp"
#include "CtensorB_array.hpp"
// #include "Dobject.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
#include "CtensorObj_helpers.hpp"
//#include "GenericOperators.hpp"

//#include "cell_ops/CtensorA_setIdentity_cop.hpp"


namespace cnine{


  class CtensorArray: public CNINE_CTENSORARRAY_IMPL{
  public:

    using CNINE_CTENSORARRAY_IMPL::CNINE_CTENSORARRAY_IMPL; 


  public: // ---- Constructors -----------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorArray(const Gdims& _adims, const Gdims& _cdims, const FILLTYPE& dummy, const device& _dev):
      CNINE_CTENSORARRAY_IMPL(_adims,_cdims,dummy,_dev.id()){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static CtensorArray raw(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArray(_adims,_dims,fill::raw,_dev);}
    static CtensorArray raw(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return CtensorArray(_adims,_dims,fill::raw,_dev.id());}

    static CtensorArray zero(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArray(_adims,_dims,fill::zero,_dev);}
    static CtensorArray zero(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return CtensorArray(_adims,_dims,fill::zero,_dev.id());}

    static CtensorArray ones(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArray(_adims,_dims,fill::ones,_dev);}
    static CtensorArray ones(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return CtensorArray(_adims,_dims,fill::ones,_dev.id());}

    //static CtensorArray identity(const Gdims& _adims, const Gdims& _dims=-1, const int _dev=0){
    //return CtensorArray(_adims,_dims,CtensorA_setIdentity_cop(),_dev);}
    //static CtensorArray identity(const Gdims& _adims, const Gdims& _dims, const device& _dev){
    //return CtensorArray(_adims,_dims,CtensorA_setIdentity_cop(),_dev.id());}
    //static CtensorArray identity(const Gdims& _adims, const Gdims& _dims, const device& _dev){
    //return CtensorArray(_adims,_dims,-1,CtensorA_setIdentity_cop(),_dev.id());}


    static CtensorArray sequential(const Gdims& _adims, const Gdims& _dims=-1, const int _dev=0){
      return CtensorArray(_adims,_dims,fill::sequential,_dev);}
    static CtensorArray sequential(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return CtensorArray(_adims,_dims,fill::sequential,_dev.id());}

    static CtensorArray gaussian(const Gdims& _adims, const Gdims& _dims=-1, const int _dev=0){
      return CtensorArray(_adims,_dims,fill::gaussian,_dev);}
    static CtensorArray gaussian(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return CtensorArray(_adims,_dims,fill::gaussian,_dev.id());}
    

  public: // ---- Copying ------------------------------------------------------------------------------------

    
    CtensorArray(const CtensorArray& x):
      CNINE_CTENSORARRAY_IMPL(x){};
      
    CtensorArray(const CtensorArray& x, const int _dev):
      CNINE_CTENSORARRAY_IMPL(x,_dev){};
      
    CtensorArray(const CtensorArray& x, const device& _dev):
      CNINE_CTENSORARRAY_IMPL(x,_dev.id()){};
      
    CtensorArray(CtensorArray&& x):
      CNINE_CTENSORARRAY_IMPL(std::move(x)){};

    CtensorArray& operator=(const CtensorArray& x){
      CNINE_CTENSORARRAY_IMPL::operator=(x);
      return *this;
    }

    CtensorArray& operator=(CtensorArray&& x){
      CNINE_CTENSORARRAY_IMPL::operator=(std::move(x));
      return *this;
    }
    
    template<typename FILLTYPE>
    CnineObject* spawn(const FILLTYPE& fill) const{
      return new CtensorArray(adims,cdims,fill,dev);
    }

    template<typename FILLTYPE>
    CnineObject* spawn(const  Gdims _adims,const FILLTYPE& fill) const{
      return new CtensorArray(_adims,cdims,fill,dev);
    }

    template<typename FILLTYPE>
    CnineObject* spawn_cell(const FILLTYPE& fill) const{
      return new CtensorObj(cdims,fill,dev);
    }


    
    CtensorArray(CtensorArray& x, const cnine::view_flag& flag):
      CNINE_CTENSORARRAY_IMPL(x,flag){}

      
  public: // ---- Conversions --------------------------------------------------------------------------------

    
    CtensorArray(const CNINE_CTENSORARRAY_IMPL& x):
      CNINE_CTENSORARRAY_IMPL(x){};
      
    CtensorArray(const Conjugate<CtensorArray>& x):
      CtensorArray(x.obj.conj()){}

    /*
    CtensorArray(const Transpose<CtensorArray>& x):
      CtensorArray(x.obj.transp()){}

    CtensorArray(const Hermitian<CtensorArray>& x):
      CtensorArray(x.obj.herm()){}

    CtensorArray(const Transpose<Conjugate<CtensorArray> >& x):
      CtensorArray(x.obj.obj.herm()){}

    CtensorArray(const Conjugate<Transpose<CtensorArray> >& x):
      CtensorArray(x.obj.obj.herm()){}
    */

  public: // ---- Transport -----------------------------------------------------------------------------------


    CtensorArray to(const device& _dev) const{
      return CtensorArray(*this,_dev);
    }

    CtensorArray to_device(const int _dev) const{
      return CtensorArray(*this,_dev);
    }


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

    Gdims get_dims() const{ 
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }
    */

    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
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


  public: // ---- Broadcasting and reductions ----------------------------------------------------------------


    CtensorArray(const Gdims& _adims, const CtensorObj& x):
      CtensorArray(_adims,x.dims,x.get_nbu(),fill::raw,x.dev){
      //cout<<"a="<<asize<<endl;
      broadcast_copy(x);
    }

    CtensorArray widen(const int ix, const int n) const{
      assert(ix<=adims.size());
      CtensorArray R(adims.insert(ix,n),cdims,nbu,fill::raw,dev);
      R.broadcast_copy(ix,*this);
      return R;
    }

    CtensorArray repeat(const int ix, const int n) const{
      assert(ix<=adims.size());
      CtensorArray R(adims.insert(ix,n),cdims,nbu,fill::raw,dev);
      R.broadcast_copy(ix,*this);
      return R;
    }

    CtensorArray reduce(const int ix) const{
      assert(ix<adims.size());
      CtensorArray R(adims.remove(ix),cdims,nbu,fill::zero,dev);
      R.add_reduce(*this,ix);
      return R;
    }


  public: // ---- Multiplication by scattered tensors --------------------------------------------------------
    

    CtensorArray& operator*=(const Scatter<CtensorObj>& x){
      inplace_scatter_times(x.obj);
      return *this; 
    }

    CtensorArray& operator/=(const Scatter<CtensorObj>& x){
      inplace_scatter_div(x.obj);
      return *this; 
    }


  public: // ---- In-place operations ------------------------------------------------------------------------



  public: // ---- Not in-place operations --------------------------------------------------------------------


    CtensorArray conj() const{
      return CtensorArray(CNINE_CTENSORARRAY_IMPL::conj());
    }

    /*
    CtensorObj transp() const{
      return CtensorObj(CNINE_CTENSORARRAY_IMPL::transp());
    }

    CtensorObj herm() const{
      return CtensorObj(CNINE_CTENSORARRAY_IMPL::herm());
    }
    */

    CtensorArray plus(const CtensorArray& x) const{
      return CtensorArray(CNINE_CTENSORARRAY_IMPL::plus(x));
    }

    /*
    CtensorObj apply(std::function<complex<float>(const complex<float>)> fn) const{
      return CNINE_CTENSORARRAY_IMPL(*this,fn);
    }

    CtensorObj apply(std::function<complex<float>(const int i, const int j, const complex<float>)> fn) const{
      return CNINE_CTENSORARRAY_IMPL(*this,fn);
    }
    */

    CtensorArray broadcast_plus(const CtensorObj& x) const{
      return CtensorArray(CNINE_CTENSORARRAY_IMPL::plus(x));
    }

    CtensorArray broadcast_minus(const CtensorObj& x) const{
      return CtensorArray(CNINE_CTENSORARRAY_IMPL::minus(x));
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_mprod(const CtensorArray& x, const CtensorArray& y){
      add_Mprod_AA<0>(x,y);
    }

    void add_mprod_AT(const CtensorArray& x, const CtensorArray& y){
      add_Mprod_AT<0>(x,y);
    }

    void add_mprod_TA(const CtensorArray& x, const CtensorArray& y){
      add_Mprod_TA<0>(x,y);
    }

    void add_mprod_AC(const CtensorArray& x, const CtensorArray& y){
      add_Mprod_AA<2>(x,y);
    }

    void add_mprod_TC(const CtensorArray& x, const CtensorArray& y){
      add_Mprod_TA<2>(x,y);
    }

    void add_mprod_AH(const CtensorArray& x, const CtensorArray& y){
      add_Mprod_AT<2>(x,y);
    }

    void add_mprod_HA(const CtensorArray& x, const CtensorArray& y){
      add_Mprod_TA<1>(x,y);
    }


  public: // ---- Broadcast cumulative operations ------------------------------------------------------------

    
    void broadcast_add_mprod(const CtensorObj& x, const CtensorArray& y){
      broadcast_add_Mprod_AA<0>(x,y);
    }

    void broadcast_add_mprod(const CtensorArray& x, const CtensorObj& y){
      broadcast_add_Mprod_AA<0>(x,y);
    }


  public: // ---- Slices and chunks --------------------------------------------------------------------------


    /*
    CtensorArray chunk(const int ix, const int offs, const int n){
      return CNINE_CTENSORARRAY_IMPL::chunk(ix,offs,n);
    }

    CtensorArray slice(const int ix, const int offs){
      return CNINE_CTENSORARRAY_IMPL::slice(ix,offs);
    }
    */

    /*
    void add_to_slice(const int ix, const int offs, const CtensorArray& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_to_chunk(const int ix, const int offs, const CtensorArray& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_chunk_op>(hdl,x.hdl,ix,offs));
    }
    */

    /*
    void add_to_slices(const int ix, vector<const CtensorArray*> _v){
      vector<const CNINE_CTENSORARRAY_IMPL*> v(_v.size());
      for(int i=0; i<_v.size(); i++) v[i]=_v[i];
      CNINE_CTENSORARRAY_IMPL::add_to_slices(ix,v);
    }
    
    template<typename... Args>
    void add_to_slices(const int ix, const CtensorArray& x0, Args... args){
      add_to_slices(ix,const_variadic_unroller(x0,args...));
    }
    */

    /*
    void add_slice(const CtensorArray& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_add_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void set_chunk(const CtensorArray& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_set_chunk_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_chunk(const CtensorArray& x, const int ix, const int offs, const int n){
      replace(hdl,Cengine_engine->push<ctensor_add_chunk_op>(hdl,x.hdl,ix,offs,n));
    }
    
    CtensorArray slice(const int ix, const int offs) const{
      CtensorArray R(dims.remove(ix),fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_slice_op>(R.hdl,hdl,ix,offs));
      return R;
    }

    CtensorArray chunk(const int ix, const int offs, const int n=1) const{
      Gdims Rdims=dims;
      Rdims[ix]=n;
      CtensorArray R(Rdims,fill::zero);
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
      add(x,g);
      add_cconj(x,g);
    }


  // ---- In-place operators ---------------------------------------------------------------------------------


    CtensorArray& operator+=(const CtensorArray& y){
      add(y);
      return *this;
    }

    CtensorArray& operator-=(const CtensorArray& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


    CtensorArray operator+(const CtensorArray& y) const{
      CtensorArray R(*this);
      R.add(y);
      return R;
    }

    CtensorArray operator-(const CtensorArray& y) const{
      CtensorArray R(*this);
      R.subtract(y);
      return R;
    }

    //CtensorObj operator*(const CscalarObj& c) const{
    //CtensorObj R(get_dims(),get_nbu(),fill::zero);
    //R.add(*this,c);
    //return R;
    //}

    CtensorArray operator*(const CtensorArray& y) const{
      int I=cdims.combined(0,cdims.k()-1);
      int J=y.cdims.combined(1,y.cdims.k());
      CtensorArray R(adims,cnine::dims(I,J),nbu,fill::zero,dev);
      R.add_mprod(*this,y);
      return R;
    }

    /*
    CtensorArray operator*(const Transpose<CtensorArray>& y) const{
      int I=dims.combined(0,dims.k()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.k()-1);
      CtensorArray R(a{I,J},fill::zero);
      R.add_mprod_AT(*this,y.obj);
      return R;
    }
    */

  public: // ---- Normalization ------------------------------------------------------------------------------


    /*
    CtensorObj col_norms() const{
      Gdims _dims=get_dims();
      CtensorObj R(_dims.remove(_dims.size()-2),get_nbu(),fill::zero,dev);
      R.add_col_norms(*this);
      return R;
    }

    CtensorObj divide_cols(const CtensorObj& N) const{
      return CtensorObj(CNINE_CTENSORARRAY_IMPL::divide_cols(N)); 
    }
    
    CtensorObj normalize_cols() const{
      return CtensorObj(CNINE_CTENSORARRAY_IMPL::normalize_cols()); 
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::CtensorObj";
    }

    string describe() const{
      return "Ctensor"+dims.str();
    } 

    string str(const string indent="") const{
      return CNINE_CTENSORARRAY_IMPL::str(indent);
    }

    string repr() const{
      return "<cnine::CtensorArray"+adims.str()+cdims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const CtensorArray& x){
      stream<<x.str(); return stream;}

    //friend ostream& operator<<(ostream& stream, CtensorObj x){
    //stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------


  inline CtensorArray& asCtensorArray(CnineObject* x){
    assert(x); 
    if(!dynamic_cast<CtensorArray*>(x))
      cerr<<"cnine error: object is of type "<<x->classname()<<" instead of CtensorArray."<<endl;
    assert(dynamic_cast<CtensorArray*>(x));
    return static_cast<CtensorArray&>(*x);
  }



  // ---------------------------------------------------------------------------------------------------------


}


#endif

    /*
    CtensorArray(const Gdims& _adims, const Gdims& _dims, const int _dev=0):
      CNINE_CTENSORARRAY_IMPL(_adims,_dims,fill_raw(),_dev){}

    CtensorArray(const Gdims& _adims, const Gdims& _dims, const int _nbu, const int _dev):
      CNINE_CTENSORARRAY_IMPL(_adims,_dims,_nbu,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorArray(const Gdims& _adims, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      CNINE_CTENSORARRAY_IMPL(_adims,_dims,fill,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorArray(const Gdims& _adims, const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      CNINE_CTENSORARRAY_IMPL(_adims,_dims,_nbu,fill,_dev){}
    
    CtensorArray(const Gdims& _adims, const Gdims& _dims, std::function<complex<float>(const int i, const int j)> fn):
      CNINE_CTENSORARRAY_IMPL(_adims,_dims,fn){}
    */

    //public: // ---- Public Constructors ------------------------------------------------------------------------
