/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineDtensor
#define _CnineDtensor

#include "Cnine_base.hpp"
#include "NamedTypes.hpp"
//#include "TensorBase.hpp"
#include "Tensor.hpp"
#include "Ltensor.hpp"


namespace cnine{


  class Dtensor{
  public:

    dtype_enum dtype;
    TensorBase* T;

    ~Dtensor(){
      delete T;
    }



  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      //int b=0;
      //Gdims gdims;
      Gdims cdims;
      int fcode=0;
      int dev=0;
      //Dtype dtype=Dtype::dfloat;
      dtype_enum _dtype=dfloat;
    };      

    template<typename... Args>
    Dtensor(const Gdims& _cdims, const Args&... args){
      vparams v;
      v.cdims=_cdims;
      unroller(v,args...);
      if(v._dtype==dint) T=new Tensor<int>(v.cdims,v.fcode,v.dev);
      if(v._dtype==dfloat) T=new Tensor<float>(v.cdims,v.fcode,v.dev);
      if(v._dtype==ddouble) T=new Tensor<double>(v.cdims,v.fcode,v.dev);
      //if(v._dtype==cdint) T=new Tensor<complex<int> >(v.cdims,v.fcode,v.dev);
      if(v._dtype==dcfloat) T=new Tensor<complex<float> >(v.cdims,v.fcode,v.dev);
      //if(v._dtype==cddouble) T=new Tensor<complex<double> >(v.cdims,v.fcode,v.dev);
      dtype=v._dtype;
    }

  //template<typename... Args>
  //void unroller(vparams& v, const cnine::BatchArgument& x, const Args&... args){
  //  v.b=x.get(); unroller(v, args...);}

  //template<typename... Args>
  //void unroller(vparams& v, const cnine::GridArgument& x, const Args&... args){
  //  v.gdims=x.get(); unroller(v, args...);}

  //template<typename... Args>
  //void unroller(vparams& v, const cnine::DimsArgument& x, const Args&... args){
  //v.cdims=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::FillArgument& x, const Args&... args){
      v.fcode=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::DeviceArgument& x, const Args&... args){
      v.dev=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::DtypeArgument& x, const Args&... args){
      v._dtype=x.get(); unroller(v, args...);}

    void unroller(vparams& v){}


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Dtensor(const TensorView<float>& x){
      T=new Tensor<float>(x);
      dtype=dfloat;
    }

    Dtensor(const TensorView<complex<float> >& x){
      T=new Tensor<complex<float> >(x);
      dtype=dcfloat;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    dtype_enum type() const{
      if(dynamic_cast<TensorView<int>*>(T)) return dint;
      if(dynamic_cast<TensorView<float>*>(T)) return dfloat;
      if(dynamic_cast<TensorView<double>*>(T)) return ddouble;
      //if(dynamic_cast<TensorView<complex<int> >*>(T)) return ddint;
      if(dynamic_cast<TensorView<complex<float> >*>(T)) return dcfloat;
      //if(dynamic_cast<TensorView<complex<double> >*>(T)) return cddouble;
      return dint;
    }

    template<typename TYPE> 
    TensorView<TYPE>& tview() const{
      CNINE_ASSRT(dynamic_cast<TensorView<TYPE>*>(T));
      return *dynamic_cast<TensorView<TYPE>*>(T);
    }


  public: // ---- float Getters -----------------------------------------------------------------------------


    int get_int(const int i0) const{
      if (type()==dint) return tview<int>()(i0);
      if (type()==dfloat) return tview<float>()(i0);
      if (type()==ddouble) return tview<double>()(i0);
      return 0;
    }

    int get_int(const int i0, const int i1) const{
      if (type()==dint) return tview<int>()(i0,i1);
      if (type()==dfloat) return tview<float>()(i0,i1);
      if (type()==ddouble) return tview<double>()(i0,i1);
      return 0;
    }

    int get_int(const int i0, const int i1, const int i2) const{
      if (type()==dint) return tview<int>()(i0,i1,i2);
      if (type()==dfloat) return tview<float>()(i0,i1,i2);
      if (type()==ddouble) return tview<double>()(i0,i1,i2);
      return 0;
    }


  public: // ---- float getters -----------------------------------------------------------------------------


    float get_float(const int i0) const{
      if (type()==dint) return tview<int>()(i0);
      if (type()==dfloat) return tview<float>()(i0);
      if (type()==ddouble) return tview<double>()(i0);
      return 0;
    }

    float get_float(const int i0, const int i1) const{
      if (type()==dint) return tview<int>()(i0,i1);
      if (type()==dfloat) return tview<float>()(i0,i1);
      if (type()==ddouble) return tview<double>()(i0,i1);
      return 0;
    }

    float get_float(const int i0, const int i1, const int i2) const{
      if (type()==dint) return tview<int>()(i0,i1,i2);
      if (type()==dfloat) return tview<float>()(i0,i1,i2);
      if (type()==ddouble) return tview<double>()(i0,i1,i2);
      return 0;
    }


  public: // ---- double getters ----------------------------------------------------------------------------


    double get_double(const int i0) const{
      if (type()==dint) return tview<int>()(i0);
      if (type()==dfloat) return tview<float>()(i0);
      if (type()==ddouble) return tview<double>()(i0);
      return 0;
    }

    double get_double(const int i0, const int i1) const{
      if (type()==dint) return tview<int>()(i0,i1);
      if (type()==dfloat) return tview<float>()(i0,i1);
      if (type()==ddouble) return tview<double>()(i0,i1);
      return 0;
    }

    double get_double(const int i0, const int i1, const int i2) const{
      if (type()==dint) return tview<int>()(i0,i1,i2);
      if (type()==dfloat) return tview<float>()(i0,i1,i2);
      if (type()==ddouble) return tview<double>()(i0,i1,i2);
      return 0;
    }


  public: // ---- cfloat getters -----------------------------------------------------------------------------


    complex<float> get_cfloat(const int i0) const{
      if (type()==dint) return tview<int>()(i0);
      if (type()==dfloat) return tview<float>()(i0);
      if (type()==ddouble) return tview<double>()(i0);
      if (type()==dcfloat) return tview<complex<float> >()(i0);
      return 0;
    }

    complex<float> get_cfloat(const int i0, const int i1) const{
      if (type()==dint) return tview<int>()(i0,i1);
      if (type()==dfloat) return tview<float>()(i0,i1);
      if (type()==ddouble) return tview<double>()(i0,i1);
      if (type()==dcfloat) return tview<complex<float> >()(i0,i1);
      return 0;
    }

    complex<float> get_cfloat(const int i0, const int i1, const int i2) const{
      if (type()==dint) return tview<int>()(i0,i1,i2);
      if (type()==dfloat) return tview<float>()(i0,i1,i2);
      if (type()==ddouble) return tview<double>()(i0,i1,i2);
      if (type()==dcfloat) return tview<complex<float> >()(i0,i1,i2);
      return 0;
    }


  public: // ---- setters ------------------------------------------------------------------------------------


    template<typename TYPE>
    void set(const int i0, const TYPE v) const{
      if (type()==dint) return tview<int>().set(i0,v);
      if (type()==dfloat) return tview<float>().set(i0,v);
      if (type()==ddouble) return tview<double>().set(i0,v);
      if (type()==dcfloat) return tview<complex<float> >().set(i0,v);
    }

    template<typename TYPE>
    void set(const int i0, const int i1, const TYPE v) const{
      if (type()==dint) return tview<int>().set(i0,i1,v);
      if (type()==dfloat) return tview<float>().set(i0,i1,v);
      if (type()==ddouble) return tview<double>().set(i0,i1,v);
      if (type()==dcfloat) return tview<complex<float> >().set(i0,i1,v);
    }

    template<typename TYPE>
    void set(const int i0, const int i1, const int i2, const TYPE v) const{
      if (type()==dint) return tview<int>().set(i0,i1,i2,v);
      if (type()==dfloat) return tview<float>().set(i0,i1,i2,v);
      if (type()==ddouble) return tview<double>().set(i0,i1,i2,v);
      if (type()==dcfloat) return tview<complex<float> >().set(i0,i1,i2,v);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Dtensor";
    }

    string str(const string indent="") const{
      return T->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Dtensor& x){
      stream<<x.str(); return stream;
    }


  };

}


#endif 
    //template<typename TYPE> 
    //TensorView<TYPE>& tview(){
    //CNINE_ASSRT(dynamic_cast<TensorView<TYPE>*>(T));
    //return *dynamic_cast<TensorView<TYPE>*>(T);
    //}

  //using BatchArgument=NamedType<int, struct BatchArgumentTag>;
  //using GridArgument=NamedType<Gdims, struct GridArgumentTag>;
  //using DimsArgument=NamedType<Gdims, struct DimsArgumentTag>;
  //using ChannelsArgument=NamedType<int, struct ChannelsArgumentTag>;
  //static const BatchArgument::argument batch;
  //static const GridArgument::argument grid;
  //static const DimsArgument::argument cdims;
  //static const ChannelsArgument::argument channels;

  /*
  using FillArgument=NamedType<int, struct FillArgumentTag>;
  using DeviceArgument=NamedType<int, struct DeviceArgumentTag>;
  using DtypeArgument=NamedType<dtype_enum, struct DtypeArgumentTag>;
  
  static const FillArgument::argument filltype;
  static const DeviceArgument::argument device;
  static const DtypeArgument::argument dtype;
  */
