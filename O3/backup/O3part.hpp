/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElibO3part
#define _GElibO3part

#include "Gpart.hpp"
#include "O3group.hpp"
#include "O3index.hpp"
#include "O3type.hpp"


namespace GElib{

  extern O3CGbank O3_CGbank;


  #ifdef _WITH_CUDA
  template<typename TYPE> class O3part;
  void O3part_addCGproduct_cu(const O3part<float>& r, const O3part<float>& x, const O3part<float>& y, const int offs, const cudaStream_t& stream);
  void O3part_addCGproduct_back0_cu(const O3part<float>& r, const O3part<float>& x, const O3part<float>& y, const int offs, const cudaStream_t& stream);
  void O3part_addCGproduct_back1_cu(const O3part<float>& r, const O3part<float>& x, const O3part<float>& y, const int offs, const cudaStream_t& stream);
  // void O3part_addDiagCGproduct_cu(const O3part<float>& r, const O3part<float>& x, const O3part<float>& y, const int offs, const cudaStream_t& stream);
  // void O3part_addDiagCGproduct_back0_cu(const O3part<float>& r, const O3part<float>& x, const O3part<float>& y, const int offs, const cudaStream_t& stream);
  // void O3part_addDiagCGproduct_back1_cu(const O3part<float>& r, const O3part<float>& x, const O3part<float>& y, const int offs, const cudaStream_t& stream);
  #endif

  

  template<typename TYPE>
  class O3part: public Gpart<O3part<TYPE>,complex<TYPE> >{
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

    //O3part(const int _b, const int _l, const int _p, const int _nc, const int _fcode=0, const int _dev=0):
    //BASE(_b,2*_l+1,_nc,_fcode,_dev),
    //ix(_l,_p){}

    //O3part(const int _b, const Gdims& _gdims, const int _l, const int _p, const int _nc, const int _fcode=0, const int _dev=0):
    //BASE(_b,_gdims,2*_l+1,_nc,_fcode,_dev),
    //ix(_l,_p){}

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
      return O3part(BASE::zeros_like(),ix);
    }

    O3part zeros_like(const O3index& _ix) const{
      return O3part(_ix,BASE::zeros_like(2*_ix.getl()+1));
    }

    O3part zeros_like(const O3index& _ix, const int n) const{
      return O3part(ix,BASE::zeros_like(2*_ix.getl()+1,n));
    }

    static O3part zeros_like(const O3index& _ix, const cnine::TensorView<TYPE>& x, const int fcode=0, const int dev=0){
      int d=x.ndims();
      int nc=cnine::ifthen(d>1,x.dims(-1),1);
      int b=cnine::ifthen(d>2,x.dims[0],1);
      Gdims gdims=cnine::ifthen(d>3,x.dims.chunk(1,d-3),Gdims());
      return O3part(_ix,b,gdims,nc,fcode,dev);
    }

    static O3part zeros_like(const O3part& x, const Gdims& gdims){
      return O3part(x.ix, x.getb(),gdims,x.getn(),0,x.get_dev());
    }


  public: // ---- Named constructors -------------------------------------------------------------------------

    /*
    static O3part spharm(const int l, const cnine::TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==3);
      CNINE_ASSRT(x.dim(2)==3);
      O3part R=zeros_like(l,x);
      R.add_spharm(x);
      return R;
      }
    */

  public: // ---- Copying ------------------------------------------------------------------------------------



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
      return ix.getl();
    }

    int getl() const{
      return ix.getl();
    }

    int getp() const{
      return ix.getp();
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    RTENSOR get_CGmatrix(const O3part& x, const O3part& y){
      return SO3_CGbank.get<TYPE>(x.getl(),y.getl(),getl());
    }

    static void add_CGproduct_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      const int l=(r.dims[0]-1)/2; 
      const int l1=(x.dims[0]-1)/2; 
      const int l2=(y.dims[0]-1)/2;
      const int N1=x.dims[1];
      const int N2=y.dims[1];
      for(int n1=0; n1<N1; n1++){
	for(int n2=0; n2<N2; n2++){
	  for(int m1=-l1; m1<=l1; m1++){
	    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	      r.inc(m1+m2+l,offs+n2,C(m1+l1,m2+l2)*x(m1+l1,n1)*y(m2+l2,n2));
	    }
	  }
	}
	offs+=N2;
      }
    }

    static void add_CGproduct_back0_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      const int l=(r.dims[0]-1)/2; 
      const int l1=(x.dims[0]-1)/2; 
      const int l2=(y.dims[0]-1)/2;
      const int N1=x.dims[1];
      const int N2=y.dims[1];
      for(int n1=0; n1<N1; n1++){
	for(int n2=0; n2<N2; n2++){
	  for(int m1=-l1; m1<=l1; m1++){
	    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	      x.inc(m1+l1,n1,C(m1+l1,m2+l2)*r(m1+m2+l,offs+n2)*std::conj(y(m2+l2,n2)));
	    }
	  }
	}
	offs+=N2;
      }
    }

    static void add_CGproduct_back1_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      const int l=(r.dims[0]-1)/2; 
      const int l1=(x.dims[0]-1)/2; 
      const int l2=(y.dims[0]-1)/2;
      const int N1=x.dims[1];
      const int N2=y.dims[1];
      for(int n1=0; n1<N1; n1++){
	for(int n2=0; n2<N2; n2++){
	  for(int m1=-l1; m1<=l1; m1++){
	    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	      y.inc(m2+l2,n2,C(m1+l1,m2+l2)*r(m1+m2+l,offs+n2)*std::conj(x(m1+l1,n1)));
	    }
	  }
	}
	offs+=N2;
      }
    }

    static void add_CGproduct_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      CUDA_STREAM(O3part_addCGproduct_cu(r,x,y,_offs,stream));
    }

    static void add_CGproduct_back0_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      CUDA_STREAM(O3part_addCGproduct_back0_cu(r,x,y,_offs,stream));
    }

    static void add_CGproduct_back1_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      CUDA_STREAM(O3part_addCGproduct_back1_cu(r,x,y,_offs,stream));
    }



  public: // ---- Diag CG-products --------------------------------------------------------------------------------

    
    static void add_DiagCGproduct_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      const int l=(r.dims[0]-1)/2; 
      const int l1=(x.dims[0]-1)/2; 
      const int l2=(y.dims[0]-1)/2;
      const int N1=x.dims[1];
      const int N2=y.dims[1];
      GELIB_ASSRT(N1==N2);
      for(int n=0; n<N1; n++){
	for(int m1=-l1; m1<=l1; m1++){
	  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	    r.inc(m1+m2+l,offs+n,C(m1+l1,m2+l2)*x(m1+l1,n)*y(m2+l2,n));
	  }
	}
      }
    }

    static void add_DiagCGproduct_back0_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      const int l=(r.dims[0]-1)/2; 
      const int l1=(x.dims[0]-1)/2; 
      const int l2=(y.dims[0]-1)/2;
      const int N1=x.dims[1];
      const int N2=y.dims[1];
      GELIB_ASSRT(N1==N2);
      for(int n=0; n<N1; n++){
	for(int m1=-l1; m1<=l1; m1++){
	  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	    x.inc(m1+l1,n,C(m1+l1,m2+l2)*r(m1+m2+l,offs+n)*std::conj(y(m2+l2,n)));
	  }
	}
      }
    }

    static void add_DiagCGproduct_back1_kernel(const TENSOR& r, const TENSOR& x, const TENSOR& y, const RTENSOR& C, int offs=0){
      const int l=(r.dims[0]-1)/2; 
      const int l1=(x.dims[0]-1)/2; 
      const int l2=(y.dims[0]-1)/2;
      const int N1=x.dims[1];
      const int N2=y.dims[1];
      GELIB_ASSRT(N1==N2);
      for(int n=0; n<N1; n++){
	for(int m1=-l1; m1<=l1; m1++){
	  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	    y.inc(m2+l2,n,C(m1+l1,m2+l2)*r(m1+m2+l,offs+n)*std::conj(x(m1+l1,n)));
	  }
	}
      }
    }

    static void add_DiagCGproduct_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      //CUDA_STREAM(O3part_addDiagCGproduct_cu(r,x,y,_offs,stream));
    }

    static void add_DiagCGproduct_back0_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      //CUDA_STREAM(O3part_addDiagCGproduct_back0_cu(r,x,y,_offs,stream));
    }

    static void add_DiagCGproduct_back1_dev(const O3part& r, O3part x, O3part y, const int _offs=0){
      //CUDA_STREAM(O3part_addDiagCGproduct_back1_cu(r,x,y,_offs,stream));
    }


  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    void add_spharm(const RTENSOR& x){
      O3part_addSpharmFn<TYPE>()(*this,x);
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

