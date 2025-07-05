// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibO3part
#define _GElibO3part

#include "Gpart.hpp"
#include "O3group.hpp"
#include "O3index.hpp"
#include "O3type.hpp"
#include "SO3part.hpp"
#include "SO3CGbank.hpp"


namespace GElib{


  

  template<typename TYPE>
  class O3part: public Gpart<O3part<TYPE>,complex<TYPE> >, public GpartBase{
  public:

    typedef Gpart<O3part<TYPE>,complex<TYPE> > BASE;
    typedef cnine::TensorView<complex<TYPE> > TENSOR;
    typedef cnine::TensorView<TYPE> RTENSOR;

    typedef O3group GROUP;
    typedef O3index IRREP_IX; 
    typedef O3type GTYPE;

    static constexpr int null_ix=-1;

    typedef cnine::Gdims Gdims;

    using TENSOR::get_dev;
    using TENSOR::ndims;
    using TENSOR::dim;
    using TENSOR::dims;
    using TENSOR::inc;
    using TENSOR::dtype_str;

    using BASE::unroller;
    //using BASE::zeros_like;
    using BASE::getb;
    using BASE::getn;
    //using BASE::dominant_batch;
    //using BASE::dominant_gdims;
    //using BASE::co_promote;
    //using BASE::fuse_and_co_promote;
    //using BASE::canonicalize;
    //using BASE::co_canonicalize;

    O3index ix;


  public: // ---- Constructors -------------------------------------------------------------------------------


    O3part(){}

    O3part(const O3index& _ix, const int _b, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,2*_ix.getl()+1,_nc,_fcode,_dev),
      ix(_ix){}

    O3part(const O3index& _ix, const int _b, const Gdims& _gdims, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,_gdims,2*_ix.getl()+1,_nc,_fcode,_dev),
      ix(_ix){}

    O3part(const O3index& _ix, const BASE& x):
      BASE(x),
      ix(_ix){}


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    template<typename ARG0, typename... Args, 
	     typename = typename std::enable_if<
    std::is_same<IrrepArgument, ARG0>::value || 
    std::is_same<cnine::BatchArgument, ARG0>::value ||
    std::is_same<cnine::GridArgument, ARG0>::value ||
    std::is_same<cnine::ChannelsArgument, ARG0>::value ||
    std::is_same<cnine::FillArgument, ARG0>::value ||
    std::is_same<cnine::DeviceArgument, ARG0>::value, ARG0>::type>
    O3part(const ARG0& arg0, const Args&... args){
      typename BASE::vparams v;
      unroller(v,arg0,args...);
      if(v.ell.has_value()==false) 
	throw std::invalid_argument("GElib error: constructor of O3part must have an irrep argument.");
      ix=any_cast<O3index>(v.ell);
      BASE::reset(v.b,v.gdims,2*ix.getl()+1,v.nc,v.fcode,v.dev);
    }


  public: // ---- Factory methods -------------------------------------------------------------------------------------


    O3part zeros_like() const{
      return O3part(get_ix(),BASE::zeros_like());
    }

    O3part zeros_like(const O3index& _ix) const{
      return O3part(get_ix(),BASE::zeros_like(2*_ix.getl()+1));
    }

    O3part zeros_like(const O3index& _ix, const int n) const{
      return O3part(get_ix(),BASE::zeros_like(2*_ix.getl()+1,n));
    }

    static O3part zeros_like(const O3index& _ix, const cnine::TensorView<TYPE>& x, const int fcode=0, const int dev=0){
      int d=x.ndims();
      int nc=cnine::ifthen(d>1,x.dims(-1),1);
      int b=cnine::ifthen(d>2,x.dims[0],1);
      Gdims gdims=cnine::ifthen(d>3,x.dims.chunk(1,d-3),Gdims());
      return O3part(_ix,b,gdims,nc,fcode,dev);
    }

    static O3part zeros_like(const O3part& x, const Gdims& gdims){
      return O3part(x.get_ix(),x.getb(),gdims,x.getn(),0,x.get_dev());
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static O3part spharm(const int l, const cnine::TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==3);
      CNINE_ASSRT(x.dim(2)==3);
      O3part R=zeros_like(O3index(l,1-2*(l%2)),x);
      R.add_spharm(x);
      return R;
    }

    static O3part spharm(const O3index& _ix, const cnine::TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==3);
      CNINE_ASSRT(x.dim(2)==3);
      CNINE_ASSRT(_ix.getp()==1-2*(_ix.getl()%2));
      O3part R=zeros_like(_ix,x);
      R.add_spharm(x);
      return R;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    O3part(const O3part<TYPE>& x):
      BASE(x),
      ix(x.ix){}


  public: // ---- Conversions ---------------------------------------------------------------------------------

    /*
    O3part(const TENSOR& x):
      BASE(x){
      GELIB_ASSRT(ndims()>=3);
      GELIB_ASSRT(dims(-2)%2==1);
    }

    O3part like(const TENSOR& x) const{
      GELIB_ASSRT(ndims()>=3);
      GELIB_ASSRT(dims(-2)==x.dims(-2));
      return O3part(x);
    }
    */

  public: // ---- Transport -----------------------------------------------------------------------------------


    O3part(const O3part& x, const int _dev):
      O3part(x.ix,TENSOR(x,_dev)){}


  public: // ---- Access -------------------------------------------------------------------------------------

    
    O3index get_ix() const{
      return ix;
    }

    int getl() const{
      return ix.getl();
    }

    int getp() const{
      return ix.getp();
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    RTENSOR get_CGmatrix(const O3part& x, const O3part& y){
      GELIB_ASSRT(x.getp()*y.getp()==getp());
      return SO3_CGbank.get<TYPE>(x.getl(),y.getl(),getl());
    }

    static void add_CGproduct_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      SO3part<TYPE>::add_CGproduct_kernel(r,x,y,C,offs);
    }

    static void add_CGproduct_back0_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      SO3part<TYPE>::add_CGproduct_back0_kernel(r,x,y,C,offs);
    }

    static void add_CGproduct_back1_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      SO3part<TYPE>::add_CGproduct_back1_kernel(r,x,y,C,offs);
    }

    static void add_CGproduct_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      SO3part<TYPE>::add_CGproduct_dev(r,x,y,_offs);
    }

    static void add_CGproduct_back0_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      SO3part<TYPE>::add_CGproduct_back0_dev(r,x,y,_offs);
    }

    static void add_CGproduct_back1_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      SO3part<TYPE>::add_CGproduct_back1_dev(r,x,y,_offs);
    }


  public: // ---- Diag CG-products --------------------------------------------------------------------------------


    static void add_DiagCGproduct_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      SO3part<TYPE>::add_DiagCGproduct_kernel(r,x,y,C,offs);
    }

    static void add_DiagCGproduct_back0_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      SO3part<TYPE>::add_DiagCGproduct_back0_kernel(r,x,y,C,offs);
    }

    static void add_DiagCGproduct_back1_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      SO3part<TYPE>::add_DiagCGproduct_back1_kernel(r,x,y,C,offs);
    }

    static void add_DiagCGproduct_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      SO3part<TYPE>::add_DiagCGproduct_dev(r,x,y,_offs);
    }

    static void add_DiagCGproduct_back0_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      SO3part<TYPE>::add_DiagCGproduct_back0_dev(r,x,y,_offs);
    }

    static void add_DiagCGproduct_back1_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      SO3part<TYPE>::add_DiagCGproduct_back1_dev(r,x,y,_offs);
    }


  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    void add_spharm(const RTENSOR& x){
      GELIB_ASSRT(getp()==1-2*(getl()%2));
      SO3part<TYPE>::add_spharm(x);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::O3part";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<O3part<"<<dtype_str()<<">:";
      if(BASE::is_batched()) oss<<" b="<<BASE::getb()<<",";
      if(BASE::is_grid()) oss<<" grid="<<BASE::gdims()<<",";
      oss<<" l="<<getl()<<",";
      oss<<" nc="<<getn()<<",";
      if(get_dev()>0) oss<<" device="<<get_dev()<<",";
      oss<<"\b>";
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const O3part& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 

