/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _Cnine_Mtensor
#define _Cnine_Mtensor

#include "Cengine_base.hpp"
#include "ExprTemplates.hpp"
#include "Tensor.hpp"

#include "mtensor_ops.hpp"
#include "mtensor_add_ops.hpp"
#include "mtensor_add_mprod_ops.hpp"
#include "mtensor_cumulative_ops.hpp"
#include "mtensor_constructor_ops.hpp"

extern Cengine::Cengine* Cengine::cengine;


namespace cnine{


  template<typename TYPE>
  class Mtensor{
  public:

    typedef Cengine::Chandle Chandle;
    typedef Cengine::Cnode Cnode;
    typedef Cengine::Cobject Cobject;

    Cengine::Cengine* engine=Cengine::cengine;

    Gdims dims;
    //int nbu=-1;
    int dev=0; 

    Chandle* hdl=nullptr;

    ~Mtensor(){
      delete hdl; 
    }

    Mtensor(){}

    Mtensor(Chandle* _hdl, const Gdims& _dims): 
      dims(_dims), hdl(_hdl){}

    Mtensor(const Gdims& _dims): dims(_dims){
      hdl=engine->push<new_mtensor_op<TYPE> >(_dims,0);
    }

    Mtensor(const Gdims& _dims, const fill_raw& fill, const int _dev=0): 
      dims(_dims), dev(_dev){
      hdl=engine->push<new_mtensor_op<TYPE> >(_dims,dev);
    }

    Mtensor(const Gdims& _dims, const fill_zero& fill, const int _dev=0): 
      dims(_dims), dev(_dev){
      hdl=engine->push<new_mtensor_zero_op<TYPE> >(_dims,dev);
    }

    Mtensor(const Gdims& _dims, const fill_ones& fill, const int _dev=0): 
      dims(_dims), dev(_dev){
      hdl=engine->push<new_mtensor_ones_op<TYPE> >(_dims,dev);
    }

    Mtensor(const Gdims& _dims, const fill_identity& fill, const int _dev=0): 
      dims(_dims), dev(_dev){
      hdl=engine->push<new_mtensor_identity_op<TYPE> >(_dims,dev);
    }

    Mtensor(const Gdims& _dims, const fill_sequential& fill, const int _dev=0): 
      dims(_dims), dev(_dev){
      hdl=engine->push<new_mtensor_sequential_op<TYPE> >(_dims,dev);
    }

    Mtensor(const Gdims& _dims, const fill_gaussian& fill, const int _dev=0): 
      dims(_dims), dev(_dev){
      hdl=engine->push<new_mtensor_gaussian_op<TYPE> >(_dims,fill.c,dev);
    }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static Mtensor<TYPE> zero(const Gdims& _dims, const int _dev=0){
      return Mtensor<TYPE>(_dims,fill_zero(),_dev);
    }

    static Mtensor<TYPE> ones(const Gdims& _dims, const int _dev=0){
      return Mtensor<TYPE>(_dims,fill_ones(),_dev);
    }

    static Mtensor<TYPE> identity(const Gdims& _dims, const int _dev=0){
      return Mtensor<TYPE>(_dims,fill_identity(),_dev);
    }

    static Mtensor<TYPE> sequential(const Gdims& _dims, const int _dev=0){
      return Mtensor<TYPE>(_dims,fill_sequential(),_dev);
    }

    static Mtensor<TYPE> randn(const Gdims& _dims, const int _dev=0){
      return Mtensor<TYPE>(_dims,fill_gaussian(),_dev);
    }

    static Mtensor<TYPE> gaussian(const Gdims& _dims, const int _dev=0){
      return Mtensor<TYPE>(_dims,fill_gaussian(),_dev);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    Mtensor(const Mtensor& x):
      dims(x.dims),
      dev(x.dev),
      hdl(engine->push<mtensor_copy_op<TYPE> >(x.hdl)){}
    
    Mtensor(Mtensor&& x):
      dims(std::move(x.dims)),
      dev(x.dev)
    {
      hdl=x.hdl;
      x.hdl=nullptr;
    }

    Mtensor& operator=(const Mtensor& x){
      dims=x.dims;
      dev=x.dev;
      delete hdl;
      hdl=engine->push<mtensor_copy_op<TYPE> >(x.hdl);
      return *this;
    }

    Mtensor& operator=(Mtensor&& x){
      dims=x.dims;
      dev=x.dev;
      delete hdl;
      hdl=x.hdl;
      x.hdl=nullptr;
      return *this;
    }
    

  public: // ---- Conversions --------------------------------------------------------------------------------

    
    Mtensor(const Conjugate<Mtensor>& x):
      Mtensor(x.obj.conj()){}

    Mtensor(const Transpose<Mtensor>& x):
      Mtensor(x.obj.transp()){}

    Mtensor(const Hermitian<Mtensor>& x):
      Mtensor(x.obj.herm()){}

    Mtensor(const Transpose<Conjugate<Mtensor> >& x):
      Mtensor(x.obj.obj.herm()){}

    Mtensor(const Conjugate<Transpose<Mtensor> >& x):
      Mtensor(x.obj.obj.herm()){}


    operator Tensor<TYPE>() const{
      engine->flush(hdl->node);
      return MTENSOR(hdl->node->obj);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int ndims() const{ 
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

    void flush() const{
      mtensor_get(hdl);
    }

    Mtensor& to_device(const int _dev){
      dev=_dev;
      replace(hdl,engine->push<mtensor_to_device_op<TYPE> >(hdl,dev));
      return *this; 
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    //void clear(){
    //replace(hdl,engine->push<mtensor_zero_op<TYPE> >(hdl));
    //}


  public: // ---- Not in-place operations --------------------------------------------------------------------


    Mtensor conj() const{
      return Mtensor(engine->push<mtensor_conj_op<TYPE> >(hdl),dims);
    }

    Mtensor transp() const{
      return Mtensor(engine->push<mtensor_transp_op<TYPE> >(hdl),dims); //TODO
    }

    Mtensor herm() const{
      return Mtensor(engine->push<mtensor_herm_op<TYPE> >(hdl),dims);
    }

    Mtensor plus(const Mtensor& x){
      return Mtensor(engine->push<mtensor_add_op<TYPE> >(hdl,x.hdl,dims),dims);
    }

    // CscalarObject mix(const CscalarObject& x){
    //assert(dims.size()==2);
    //CscalarObject r(dims[0],fill::zero);
    //engine->push<cscalar_mix_op<TYPE> >(r.hdl,hdl,x.hdl);
    //return r;
    //}

    //Mtensor mix(const Mtensor& x){
    //assert(dims.size()==2);
    //Mtensor r(dims[0],fill::zero);
    //engine->push<mtensor_mix_op<TYPE> >(r.hdl,hdl,x.hdl);
    //return r;
    //}


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const Mtensor& x){
      replace(hdl,engine->push<mtensor_add_op<TYPE> >(hdl,x.hdl,dims));
    }

    void add_conj(const Mtensor& x){
      replace(hdl,engine->push<mtensor_add_conj_op<TYPE> >(hdl,x.hdl));
    }

    void add_transp(const Mtensor& x){
      replace(hdl,engine->push<mtensor_add_transp_op<TYPE> >(hdl,x.hdl));
    }

    void add_herm(const Mtensor& x){
      replace(hdl,engine->push<mtensor_add_herm_op<TYPE> >(hdl,x.hdl));
    }

    void subtract(const Mtensor& x){
      replace(hdl,engine->push<mtensor_subtract_op<TYPE> >(hdl,x.hdl));
    }

    void add(const Mtensor& x, const float c){
      replace(hdl,engine->push<mtensor_add_times_real_op<TYPE> >(hdl,x.hdl,c));
    }

    void add(const Mtensor& x, const complex<float> c){
      replace(hdl,engine->push<mtensor_add_times_complex_op<TYPE> >(hdl,x.hdl,c));
    }

    //void add(const Mtensor& x, const RscalarObject& c){
    //replace(hdl,engine->push<mtensor_add_prod_rA_op<TYPE> >(hdl,c.hdl,x.hdl));
    //}

    //void add(const Mtensor& x, const CscalarObject& c){
    //replace(hdl,engine->push<mtensor_add_prod_cA_op<TYPE> >(hdl,c.hdl,x.hdl));
    //}

    //void add_cconj(const Mtensor& x, const CscalarObject& c){
    //replace(hdl,engine->push<mtensor_add_prod_cc_A_op<TYPE> >(hdl,c.hdl,x.hdl));
    //}
   
    //void add_conj(const Mtensor& x, const CscalarObject& c){
    //replace(hdl,engine->push<mtensor_add_prod_c_Ac_op<TYPE> >(hdl,c.hdl,x.hdl));
    //}

    
    void add_plus(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_op<TYPE> >(hdl,x.hdl,dims));
      replace(hdl,engine->push<mtensor_add_op<TYPE> >(hdl,y.hdl,dims));
    }

    void add_minus(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_op<TYPE> >(hdl,x.hdl,dims));
      replace(hdl,engine->push<mtensor_subtract_op<TYPE> >(hdl,y.hdl));
    }


    void add_mprod(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_mprod_op<TYPE,0,0> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_mprod_AT(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_mprod_op<TYPE,2,0> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_mprod_TA(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_mprod_op<TYPE,1,0> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_mprod_AC(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_mprod_op<TYPE,0,2> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_mprod_TC(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_mprod_op<TYPE,1,2> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_mprod_AH(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_mprod_op<TYPE,2,2> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_mprod_HA(const Mtensor& x, const Mtensor& y){
      replace(hdl,engine->push<mtensor_add_mprod_op<TYPE,1,1> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }


    void add_column_norms(const Mtensor& x){
      replace(hdl,engine->push<mtensor_add_col_norms_op>(hdl,x.hdl));
    }


    void add_ReLU(const Mtensor& x, const float c=0){
      replace(hdl,engine->push<mtensor_add_ReLU_op>(hdl,x.hdl,c));
    }

    void add_ReLU_back(const Mtensor& g, const Mtensor& x, const float c=0){
      replace(hdl,engine->push<mtensor_add_ReLU_back_op>(hdl,g.hdl,x.hdl,c));
    }

    
  public: // ---- Into operations ----------------------------------------------------------------------------


    //void inp_into(const Mtensor& y, CscalarObject& R) const{
    //replace(R.hdl,engine->push<mtensor_add_inp_op>(R.hdl,hdl,y.hdl));
    //}

    //void norm2_into(CscalarObject& R) const{
    //replace(R.hdl,engine->push<mtensor_add_inp_op>(R.hdl,hdl,hdl));
    //}

    //void add_norm2_back(const CscalarObject& g, const Mtensor& x){
    //add(x,g);
    //add_conj(x,g);
    //}


  public: // ---- Operators ---------------------------------------------------------------------------------


    Mtensor& operator+=(const Mtensor& y){
      add(y);
      return *this;
    }

    Mtensor& operator-=(const Mtensor& y){
      subtract(y);
      return *this;
    }

    Mtensor operator+(const Mtensor& y){
      Mtensor R(*this);
      R.add(y);
      return R;
    }

    Mtensor operator-(const Mtensor& y){
      Mtensor R(*this);
      R.subtract(y);
      return R;
    }

    //Mtensor operator*(const CscalarObject& c){
    //Mtensor R(dims,nbu,fill::zero,dev);
    //R.add(*this,c);
    //return R;
    //}

    Mtensor operator*(const Mtensor& y){
      int I=dims.combined(0,dims.k()-1);
      int J=y.dims.combined(1,y.dims.k());
      Mtensor R({I,J},fill::zero,dev);
      R.add_mprod(*this,y);
      return R;
    }

    Mtensor operator*(const Transpose<Mtensor>& y){
      int I=dims.combined(0,dims.k()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.k()-1);
      Mtensor R({I,J},fill::zero,dev);
      R.add_mprod_AT(*this,y.obj);
      return R;
    }

    /*
    Mtensor column_norms() const{
      assert(dims.size()>=2);
      Mtensor R(dims.remove(dims.size()-2),nbu,fill::zero,dev);
      R.add_column_norms(*this);
      return R;
    }
    */

    /*
    Mtensor divide_columns(const Mtensor& N){
      return Mtensor(engine->push<mtensor_divide_cols_op>(hdl,N.hdl),dims,nbu);
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Mtensor";
    }

    string str(const string indent="") const{
      //Gtensor<complex<float> > R=mtensor_get(hdl);
      //return R.str();
      return Tensor<TYPE>(*this).str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Mtensor& x){
      stream<<x.str(); return stream;}

  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename TYPE>
  Transpose<Mtensor<TYPE> > transp(const Mtensor<TYPE>& x){
    return Transpose<Mtensor<TYPE> >(x);
  }

  template<typename TYPE>
  Conjugate<Mtensor<TYPE>> conj(const Mtensor<TYPE>& x){
    return Conjugate<Mtensor<TYPE> >(x);
  }

  template<typename TYPE>
  Hermitian<Mtensor<TYPE> > herm(const Mtensor<TYPE>& x){
    return x;
  }

  template<typename TYPE>
  Mtensor<TYPE> operator*(const Transpose<Mtensor<TYPE> >& x, const Mtensor<TYPE>& y){
    int I=x.obj.dims.combined(1,x.obj.dims.k());
    int J=y.dims.combined(1,y.dims.k());
    Mtensor<TYPE> R({I,J},fill::zero);
    R.add_mprod_TA(x.obj,y);
    return R;
  }

  //CscalarObject norm2(const Mtensor& x){
  //CscalarObject r(x.nbu,fill::zero);
  //x.norm2_into(r);
  //return r;
  //}

  template<typename TYPE>
  Mtensor<TYPE> ReLU(const Mtensor<TYPE>& x, const float c=0){
    Mtensor R(x.dims,x.nbu,fill::zero);
    R.add_ReLU(x,c);
    return R;
  }

  //CscalarObject inp(const Mtensor& x, const Mtensor& y){
  //CscalarObject r(x.nbu,fill::zero);
  //x.inp_into(y,r);
  //return r;
  //}



}


#endif

 
