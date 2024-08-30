// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGpart
#define _GElibGpart

#include "GElib_base.hpp"
#include "Ltensor.hpp"


namespace GElib{


  template<typename TYPE>
  class Gpart: public cnine::Ltensor<TYPE>{
  public:

    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    typedef cnine::Gdims Gdims;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::labels;

    using BASE::dim;
    using BASE::device;

    using BASE::bgfused_view3;

    using BASE::is_batched;
    using BASE::nbatch;

    using BASE::is_grid;
    using BASE::gdims;
    using BASE::cell;

    using BASE::cdims;


    //shared_ptr<Ggroup> G;
    //shared_ptr<GirrepIx> ix;

    ~Gpart(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    Gpart(const int _b, const int _d, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,Gdims({_d,_nc}),_fcode,_dev){}
      
    Gpart(const int _b, const Gdims& _gdims, const int _d, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,_gdims,Gdims({_d,_nc}),_fcode,_dev){}
      
    void reset(const int _b, const Gdims& _gdims, const int _d, const int _nc, const int _fcode=0, const int _dev=0){
      BASE::reset(_b,_gdims,Gdims({_d,_nc}),_fcode,_dev);
    }

      
  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int b=0;
      Gdims gdims;
      int nc=1;
      std::any ell;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    void unroller(vparams& v, const cnine::BatchArgument& x, const Args&... args){
      v.b=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::GridArgument& x, const Args&... args){
      v.gdims=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::ChannelsArgument& x, const Args&... args){
      v.nc=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const IrrepArgument& x, const Args&... args){
      v.ell=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::FillArgument& x, const Args&... args){
      v.fcode=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::DeviceArgument& x, const Args&... args){
      v.dev=x.get(); unroller(v, args...);}

    void unroller(vparams& v){}


  public: // ---- Copying ------------------------------------------------------------------------------------

    

  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nc() const{
      return dims.last();
    }

    /*
    bool is_batched() const{
      return dims.size()>2;
    }

    int getb(){
      if(!is_batched) return 0;
      return dims[0];
    }

    bool is_grid() const{
      return dims.size()>3;
    }

    int n_grid_dims(){
      return dims.size()-3;
    }

    Gdims grid_dims(){
      return dims.chunk(1,n_grid_dims());
    }
    */

    
  public: // ---- Operations ---------------------------------------------------------------------------------


 
  public: // ---- CG-products --------------------------------------------------------------------------------


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::Gpart";
    }

    string repr() const{
      ostringstream oss;
      return oss.str();
    }
    
    string to_print(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<BASE::str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gpart& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 

