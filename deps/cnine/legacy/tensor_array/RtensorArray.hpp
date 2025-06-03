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


#ifndef _CnineRtensorArray
#define _CnineRtensorArray

#include "Cnine_base.hpp"
#include "RtensorArrayA.hpp"
// #include "Dobject.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "RtensorObj.hpp"
#include "RtensorObj_helpers.hpp"
//#include "GenericOperators.hpp"

//#include "cell_ops/RtensorA_setIdentity_cop.hpp"


namespace cnine{


  class RtensorArray: public CNINE_RTENSORARRAY_IMPL{
  public:

    using CNINE_RTENSORARRAY_IMPL::CNINE_RTENSORARRAY_IMPL; 


  public: // ---- Constructors -----------------------------------------------------------------------------

    //RtensorArray(){}


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    RtensorArray(const Gdims& _adims, const Gdims& _cdims, const FILLTYPE& dummy, const device& _dev):
      CNINE_RTENSORARRAY_IMPL(_adims,_cdims,dummy,_dev.id()){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static RtensorArray zero(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return RtensorArray(_adims,_dims,fill::zero,_dev);}
    static RtensorArray zero(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return RtensorArray(_adims,_dims,fill::zero,_dev.id());}

    static RtensorArray raw(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return RtensorArray(_adims,_dims,fill::raw,_dev);}
    static RtensorArray raw(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return RtensorArray(_adims,_dims,fill::raw,_dev.id());}

    static RtensorArray ones(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return RtensorArray(_adims,_dims,fill::ones,_dev);}
    static RtensorArray ones(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return RtensorArray(_adims,_dims,fill::ones,_dev.id());}

    //static RtensorArray identity(const Gdims& _adims, const Gdims& _dims=-1, const int _dev=0){
    //return RtensorArray(_adims,_dims,RtensorA_setIdentity_cop(),_dev);}
    //static RtensorArray identity(const Gdims& _adims, const Gdims& _dims, const device& _dev){
    //return RtensorArray(_adims,_dims,RtensorA_setIdentity_cop(),_dev.id());}
    //static RtensorArray identity(const Gdims& _adims, const Gdims& _dims, const device& _dev){
    //return RtensorArray(_adims,_dims,-1,RtensorA_setIdentity_cop(),_dev.id());}


    static RtensorArray sequential(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return RtensorArray(_adims,_dims,fill::sequential,_dev);}
    static RtensorArray sequential(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return RtensorArray(_adims,_dims,fill::sequential,_dev.id());}

    static RtensorArray gaussian(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return RtensorArray(_adims,_dims,fill::gaussian,_dev);}
    static RtensorArray gaussian(const Gdims& _adims, const Gdims& _dims, const device& _dev){
      return RtensorArray(_adims,_dims,fill::gaussian,_dev.id());}
    

  public: // ---- Copying ------------------------------------------------------------------------------------

    
    RtensorArray(const RtensorArray& x):
      CNINE_RTENSORARRAY_IMPL(x){};
      
    RtensorArray(const RtensorArray& x, const int _dev):
      CNINE_RTENSORARRAY_IMPL(x,_dev){};
      
    RtensorArray(const RtensorArray& x, const device& _dev):
      CNINE_RTENSORARRAY_IMPL(x,_dev.id()){};
      
    RtensorArray(RtensorArray&& x):
      CNINE_RTENSORARRAY_IMPL(std::move(x)){};

    RtensorArray& operator=(const RtensorArray& x){
      CNINE_RTENSORARRAY_IMPL::operator=(x);
      return *this;
    }

    RtensorArray& operator=(RtensorArray&& x){
      CNINE_RTENSORARRAY_IMPL::operator=(std::move(x));
      return *this;
    }
    
    RtensorArray(RtensorArray& x, const cnine::view_flag& flag):
      CNINE_RTENSORARRAY_IMPL(x,flag){}

      
  public: // ---- Conversions --------------------------------------------------------------------------------

    
    RtensorArray(const CNINE_RTENSORARRAY_IMPL& x):
      CNINE_RTENSORARRAY_IMPL(x){};
      
    RtensorArray(const Conjugate<RtensorArray>& x):
      RtensorArray(x.obj.conj()){}

    /*
    RtensorArray(const Transpose<RtensorArray>& x):
      RtensorArray(x.obj.transp()){}

    RtensorArray(const Hermitian<RtensorArray>& x):
      RtensorArray(x.obj.herm()){}

    RtensorArray(const Transpose<Conjugate<RtensorArray> >& x):
      RtensorArray(x.obj.obj.herm()){}

    RtensorArray(const Conjugate<Transpose<RtensorArray> >& x):
      RtensorArray(x.obj.obj.herm()){}
    */

  public: // ---- Transport -----------------------------------------------------------------------------------


   RtensorArray to(const device& _dev) const{
      return RtensorArray(*this,_dev);
    }

    RtensorArray to_device(const int _dev) const{
      return RtensorArray(*this,_dev);
    }


  public: // ---- Access -------------------------------------------------------------------------------------

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

    RtensorObj get_cell(const Gindex& aix) const{
      return CNINE_RTENSORARRAY_IMPL::get_cell(aix);
    }


  public: // -------------------------------------------------------------------------------------------------

    /*
    RtensorObj& add_to_element(const Gindex& ix, const CscalarObj& v){
      replace(hdl,Cengine_engine->push<ctensor_add_to_element_op>(hdl,v.hdl,ix));
      return *this;
    }
    
    void add_element_into(CscalarObj& r, const Gindex& ix){
      replace(r.hdl,Cengine_engine->push<ctensor_add_element_op>(r.hdl,hdl,ix));
    }
    */

    /*
    int combined(const int a, const int b) const{
      return asRtensorB(hdl->node->obj).combined(a,b);
    }
    */


  public: // ---- Broadcasting and reductions ----------------------------------------------------------------


    RtensorArray(const Gdims& _adims, const RtensorObj& x):
      RtensorArray(_adims,x.dims,fill::raw,x.dev){
      //cout<<"a="<<asize<<endl;
      broadcast_copy(x);
    }

    RtensorArray widen(const int ix, const int n) const{
      assert(ix<=adims.size());
      RtensorArray R(adims.insert(ix,n),cdims,fill::raw,dev);
      R.broadcast_copy(ix,*this);
      return R;
    }

    RtensorArray repeat(const int ix, const int n) const{
      assert(ix<=adims.size());
      RtensorArray R(adims.insert(ix,n),cdims,fill::raw,dev);
      R.broadcast_copy(ix,*this);
      return R;
    }

    RtensorArray reduce(const int ix) const{
      assert(ix<adims.size());
      RtensorArray R(adims.remove(ix),cdims,fill::zero,dev);
      R.add_reduce(*this,ix);
      return R;
    }


  public: // ---- Multiplication by scattered tensors --------------------------------------------------------
    

    RtensorArray& operator*=(const Scatter<RtensorObj>& x){
      inplace_scatter_times(x.obj);
      return *this; 
    }

    RtensorArray& operator/=(const Scatter<RtensorObj>& x){
      inplace_scatter_div(x.obj);
      return *this; 
    }


  public: // ---- In-place operations ------------------------------------------------------------------------



  public: // ---- Not in-place operations --------------------------------------------------------------------


    //RtensorArray conj() const{
    //return RtensorArray(CNINE_RTENSORARRAY_IMPL::conj());
    //}

    /*
    RtensorObj transp() const{
      return RtensorObj(CNINE_RTENSORARRAY_IMPL::transp());
    }

    RtensorObj herm() const{
      return RtensorObj(CNINE_RTENSORARRAY_IMPL::herm());
    }
    */

    RtensorArray plus(const RtensorArray& x) const{
      return RtensorArray(CNINE_RTENSORARRAY_IMPL::plus(x));
    }

    /*
    RtensorObj apply(std::function<complex<float>(const complex<float>)> fn) const{
      return CNINE_RTENSORARRAY_IMPL(*this,fn);
    }

    RtensorObj apply(std::function<complex<float>(const int i, const int j, const complex<float>)> fn) const{
      return CNINE_RTENSORARRAY_IMPL(*this,fn);
    }
    */

    RtensorArray broadcast_plus(const RtensorObj& x) const{
      return RtensorArray(CNINE_RTENSORARRAY_IMPL::plus(x));
    }

    RtensorArray broadcast_minus(const RtensorObj& x) const{
      return RtensorArray(CNINE_RTENSORARRAY_IMPL::minus(x));
    }



  public: // ---- Cumulative operations ----------------------------------------------------------------------


    /*
    void add(const RtensorArray& x, const float c){
      CNINE_RTENSOR_IMPL::add(x,c);
    }
    */

    void add_mprod(const RtensorArray& x, const RtensorArray& y){
      add_Mprod_AA(x,y);
    }

    void add_mprod_AT(const RtensorArray& x, const RtensorArray& y){
      add_Mprod_AT(x,y);
    }

    void add_mprod_TA(const RtensorArray& x, const RtensorArray& y){
      add_Mprod_TA(x,y);
    }

    void add_mprod_AC(const RtensorArray& x, const RtensorArray& y){
      add_Mprod_AA(x,y);
    }

    void add_mprod_TC(const RtensorArray& x, const RtensorArray& y){
      add_Mprod_TA(x,y);
    }

    void add_mprod_AH(const RtensorArray& x, const RtensorArray& y){
      add_Mprod_AT(x,y);
    }

    void add_mprod_HA(const RtensorArray& x, const RtensorArray& y){
      add_Mprod_TA(x,y);
    }


  public: // ---- Broadcast cumulative operations ------------------------------------------------------------

    
    void broadcast_add_mprod(const RtensorObj& x, const RtensorArray& y){
      broadcast_add_Mprod_AA(x,y);
    }

    void broadcast_add_mprod(const RtensorArray& x, const RtensorObj& y){
      broadcast_add_Mprod_AA(x,y);
    }


  public: // ---- Slices and chunks --------------------------------------------------------------------------


    /*
    RtensorArray chunk(const int ix, const int offs, const int n){
      return CNINE_RTENSORARRAY_IMPL::chunk(ix,offs,n);
    }

    RtensorArray slice(const int ix, const int offs){
      return CNINE_RTENSORARRAY_IMPL::slice(ix,offs);
    }
    */

    /*
    void add_to_slice(const int ix, const int offs, const RtensorArray& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_to_chunk(const int ix, const int offs, const RtensorArray& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_chunk_op>(hdl,x.hdl,ix,offs));
    }
    */

    /*
    void add_to_slices(const int ix, vector<const RtensorArray*> _v){
      vector<const CNINE_RTENSORARRAY_IMPL*> v(_v.size());
      for(int i=0; i<_v.size(); i++) v[i]=_v[i];
      CNINE_RTENSORARRAY_IMPL::add_to_slices(ix,v);
    }
    
    template<typename... Args>
    void add_to_slices(const int ix, const RtensorArray& x0, Args... args){
      add_to_slices(ix,const_variadic_unroller(x0,args...));
    }
    */

    /*
    void add_slice(const RtensorArray& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_add_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void set_chunk(const RtensorArray& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_set_chunk_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_chunk(const RtensorArray& x, const int ix, const int offs, const int n){
      replace(hdl,Cengine_engine->push<ctensor_add_chunk_op>(hdl,x.hdl,ix,offs,n));
    }
    
    RtensorArray slice(const int ix, const int offs) const{
      RtensorArray R(dims.remove(ix),fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_slice_op>(R.hdl,hdl,ix,offs));
      return R;
    }

    RtensorArray chunk(const int ix, const int offs, const int n=1) const{
      Gdims Rdims=dims;
      Rdims[ix]=n;
      RtensorArray R(Rdims,fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_chunk_op>(R.hdl,hdl,ix,offs,n));
      return R;
    }
    */


  public: // ---- Into operations ----------------------------------------------------------------------------


    //void inp_into(CscalarObj& R, const RtensorObj& y) const{
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,y.hdl,dims));
    //}

    //void norm2_into(CscalarObj& R) const{
    //R.val=
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,hdl,dims));
    //}

    /*
    void add_norm2_back(const RscalarObj& g, const RtensorObj& x){
      add(x,g);
      add(x,g);
    }
    */


  // ---- In-place operators ---------------------------------------------------------------------------------


    RtensorArray& operator+=(const RtensorArray& y){
      add(y);
      return *this;
    }

    RtensorArray& operator-=(const RtensorArray& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


    RtensorArray operator+(const RtensorArray& y) const{
      RtensorArray R(*this);
      R.add(y);
      return R;
    }

    RtensorArray operator-(const RtensorArray& y) const{
      RtensorArray R(*this);
      R.subtract(y);
      return R;
    }

    //RtensorObj operator*(const CscalarObj& c) const{
    //RtensorObj R(get_dims(),get_nbu(),fill::zero);
    //R.add(*this,c);
    //return R;
    //}

    RtensorArray operator*(const RtensorArray& y) const{
      int I=cdims.combined(0,cdims.k()-1);
      int J=y.cdims.combined(1,y.cdims.k());
      RtensorArray R(adims,cnine::dims(I,J),fill::zero,dev);
      R.add_mprod(*this,y);
      return R;
    }

    /*
    RtensorArray operator*(const Transpose<RtensorArray>& y) const{
      int I=dims.combined(0,dims.k()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.k()-1);
      RtensorArray R(a{I,J},fill::zero);
      R.add_mprod_AT(*this,y.obj);
      return R;
    }
    */

  public: // ---- Normalization ------------------------------------------------------------------------------


    /*
    RtensorObj col_norms() const{
      Gdims _dims=get_dims();
      RtensorObj R(_dims.remove(_dims.size()-2),get_nbu(),fill::zero,dev);
      R.add_col_norms(*this);
      return R;
    }

    RtensorObj divide_cols(const RtensorObj& N) const{
      return RtensorObj(CNINE_RTENSORARRAY_IMPL::divide_cols(N)); 
    }
    
    RtensorObj normalize_cols() const{
      return RtensorObj(CNINE_RTENSORARRAY_IMPL::normalize_cols()); 
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::RtensorArray";
    }

    string describe() const{
      return "RtensorArray"+dims.str();
    } 

    string str(const string indent="") const{
      return CNINE_RTENSORARRAY_IMPL::str(indent);
    }

    string repr() const{
      return "<cnine::RtensorArray"+adims.str()+cdims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const RtensorArray& x){
      stream<<x.str(); return stream;}

    //friend ostream& operator<<(ostream& stream, RtensorObj x){
    //stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------



  // ---------------------------------------------------------------------------------------------------------


}


#endif

