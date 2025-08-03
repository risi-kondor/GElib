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

#ifndef _GElibSO3part
#define _GElibSO3part

#include "Gpart.hpp"
#include "SO3group.hpp"
#include "SO3type.hpp"
#include "SO3CGbank.hpp"
#include "TensorUtils.hpp"
//#include "SO3part_addSpharmFn.hpp"


namespace GElib{

  extern SO3CGbank SO3_CGbank;


  #ifdef _WITH_CUDA
  template<typename TYPE> class SO3part;
  void SO3part_addCGproduct_cu(const SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs, const cudaStream_t& stream);
  void SO3part_addCGproduct_back0_cu(const SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs, const cudaStream_t& stream);
  void SO3part_addCGproduct_back1_cu(const SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs, const cudaStream_t& stream);
  // void SO3part_addDiagCGproduct_cu(const SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs, const cudaStream_t& stream);
  // void SO3part_addDiagCGproduct_back0_cu(const SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs, const cudaStream_t& stream);
  // void SO3part_addDiagCGproduct_back1_cu(const SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs, const cudaStream_t& stream);
  void addSPharm_cu(const SO3part<float> r, TensorView<float> x, const cudaStream_t& stream);
  #endif

  

  template<typename TYPE>
  class SO3part: public Gpart<SO3part<TYPE>,complex<TYPE> >, public GpartBase{
  public:

    typedef Gpart<SO3part<TYPE>,complex<TYPE> > BASE;
    typedef cnine::TensorView<complex<TYPE> > TENSOR;
    typedef cnine::TensorView<TYPE> RTENSOR;

    typedef SO3group GROUP;
    typedef int IRREP_IX; 
    typedef SO3type GTYPE;

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


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3part(){}

    SO3part(const int _b, const int _l, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,2*_l+1,_nc,_fcode,_dev){}

    SO3part(const int _b, const Gdims& _gdims, const int _l, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,_gdims,2*_l+1,_nc,_fcode,_dev){}

    SO3part(const int _l, const int _b, const Gdims& _gdims, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,_gdims,2*_l+1,_nc,_fcode,_dev){}

    //SO3part(const cnine::TensorView<complex<TYPE> >& M):
    //BASE(M){}


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    template<typename ARG0, typename... Args, 
	     typename = typename std::enable_if<
    std::is_same<IrrepArgument, ARG0>::value || 
    std::is_same<cnine::BatchArgument, ARG0>::value ||
    std::is_same<cnine::GridArgument, ARG0>::value ||
    std::is_same<cnine::ChannelsArgument, ARG0>::value ||
    std::is_same<cnine::FillArgument, ARG0>::value ||
    std::is_same<cnine::DeviceArgument, ARG0>::value, ARG0>::type>
    SO3part(const ARG0& arg0, const Args&... args){
      typename BASE::vparams v;
      unroller(v,arg0,args...);
      if(v.ell.has_value()==false) 
	throw std::invalid_argument("GElib error: constructor of SO3part must have an irrep argument.");
      int ell=any_cast<int>(v.ell);
      BASE::reset(v.b,v.gdims,2*ell+1,v.nc,v.fcode,v.dev);
    }


  public: // ---- Factory methods -------------------------------------------------------------------------------------


    SO3part zeros_like() const{
      return BASE::zeros_like();
    }

    SO3part zeros_like(const int l) const{
      return BASE::zeros_like(2*l+1);
    }

    SO3part zeros_like(const int l, const int n) const{
      return BASE::zeros_like(2*l+1,n);
    }

    static SO3part zeros_like(const int l, const cnine::TensorView<TYPE>& x, const int fcode=0, const int dev=0){
      int d=x.ndims();
      int nc=cnine::ifthen(d>1,x.dims(-1),1);
      int b=cnine::ifthen(d>2,x.dims[0],1);
      Gdims gdims=cnine::ifthen(d>3,x.dims.chunk(1,d-3),Gdims());
      return SO3part(b,gdims,l,nc,fcode,dev);
    }

    static SO3part zeros_like(const SO3part& x, const Gdims& gdims){
      return SO3part(x.getb(),gdims,x.getl(),x.getn(),0,x.get_dev());
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3part spharm(const int l, const cnine::TensorView<TYPE>& x){
      GELIB_ASSRT(x.ndims()==3);
      GELIB_ASSRT(x.dim(1)==3);
      SO3part R=zeros_like(l,x);
      R.add_spharm(x);
      return R;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------



  public: // ---- Conversions ---------------------------------------------------------------------------------


    SO3part(const TENSOR& x):
      BASE(x){
      GELIB_ASSRT(ndims()>=3);
      GELIB_ASSRT(dims(-2)%2==1);
    }

    SO3part like(const TENSOR& x) const{
      GELIB_ASSRT(ndims()>=3);
      GELIB_ASSRT(dims(-2)==x.dims(-2));
      return SO3part(x);
    }


  public: // ---- Transport -----------------------------------------------------------------------------------


    SO3part(const SO3part& x, const int _dev):
      SO3part(TENSOR(x,_dev)){}


  public: // ---- Access -------------------------------------------------------------------------------------

    
    int get_ix() const{
      return (dims(-2)-1)/2;
    }

    int getl() const{
      return (dims(-2)-1)/2;
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    RTENSOR get_CGmatrix(const SO3part& x, const SO3part& y){
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

    static void add_CGproduct_dev(const SO3part& r, SO3part x, SO3part y, const int _offs=0){
      CUDA_STREAM(SO3part_addCGproduct_cu(r,x,y,_offs,stream));
    }

    static void add_CGproduct_back0_dev(const SO3part& r, SO3part x, SO3part y, const int _offs=0){
      CUDA_STREAM(SO3part_addCGproduct_back0_cu(r,x,y,_offs,stream));
    }

    static void add_CGproduct_back1_dev(const SO3part& r, SO3part x, SO3part y, const int _offs=0){
      CUDA_STREAM(SO3part_addCGproduct_back1_cu(r,x,y,_offs,stream));
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

    static void add_DiagCGproduct_dev(const SO3part& r, SO3part x, SO3part y, const int _offs=0){
      //CUDA_STREAM(SO3part_addDiagCGproduct_cu(r,x,y,_offs,stream));
    }

    static void add_DiagCGproduct_back0_dev(const SO3part& r, SO3part x, SO3part y, const int _offs=0){
      //CUDA_STREAM(SO3part_addDiagCGproduct_back0_cu(r,x,y,_offs,stream));
    }

    static void add_DiagCGproduct_back1_dev(const SO3part& r, SO3part x, SO3part y, const int _offs=0){
      //CUDA_STREAM(SO3part_addDiagCGproduct_back1_cu(r,x,y,_offs,stream));
    }


  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    void add_spharm(const RTENSOR& _x){

      RTENSOR x(_x);
      int xb=cnine::ifthen(x.ndims()>2,x.dim(0),1);
      int b=std::max(getb(),xb);
      if((xb!=getb()) && (xb!=1) && (getb()!=1)) 
	GELIB_SKIP("batch dimensions cannot be reconciled.");
      int L=getl();
      int nc=getn();

      if(get_dev()==0){
	this->template for_each_cell_multi<TYPE>(x,[&](const int b, const int g, const TENSOR& r, const RTENSOR& x){
	    for(int j=0; j<nc; j++){

	      float vx=x(0,j);
	      float vy=x(1,j);
	      float vz=x(2,j);
	      float length=sqrt(vx*vx+vy*vy+vz*vz); 
	      float len2=sqrt(vx*vx+vy*vy);

	      if(len2==0 || std::isnan(vx/len2) || std::isnan(vy/len2)){
		float a=sqrt(((float)(2*L+1))/(M_PI*4.0));
		r.inc(L,j,a); 
	      }else{
		//auto P=SO3_SPHgen(l,vz/length);

		// Recursively compute associated Legendre
		float x=vz/length;
		RTENSOR P({2*L+1,2*L+1},0,0);
		P(0,0)=1;
		float xfact=sqrt(1.0-x*x);
		for(int l=1; l<=L; l++){
		  P(l,l)=-(2.0*l-1.0)*xfact*P(l-1,l-1);
		  P(l,l-1)=(2.0*l-1.0)*P(l-1,l-1)*x;
		  for(int m=0; m<l-1; m++)
		    P(l,m)=((2.0*l-1.0)*P(l-1,m)*x-((float)(l+m-1.0))*(P(l-2,m)))/((float)(l-m));
		}

		//vector<complex<float> > phase(l+1);
		complex<float> cphi(vx/len2,vy/len2);
		complex<float> phase(sqrt((2.0*L+1.0)/(M_PI*4.0)),0);
		//for(int m=1; m<=L; m++)
		//phase[m]=cphi*sqrt(((float)(L-m))/((float)(L+m)))*phase[m-1];
	    
		for(int m=0; m<=L; m++){
		  complex<float> a=phase*P(L,m);
		  r.inc(L+m,j,a);
		  if(m>0) r.inc(L-m,j,complex<float>(1-2*(m%2))*std::conj(a));
		  if(m<L) phase*=cphi/((float)sqrt((L-m)*(L+m+1)));
		  //if(m==L-1) phase*=cphi*sqrt(((float)1.0)/((float)(L+m+1)));
		}
	      }
	    }
	  });
      }// dev==0

    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3part";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<SO3part<"<<dtype_str()<<">:";
      if(BASE::is_batched()) oss<<" b="<<BASE::getb()<<",";
      if(BASE::is_grid()) oss<<" grid="<<BASE::gdims()<<",";
      oss<<" l="<<getl()<<",";
      oss<<" nc="<<getn()<<",";
      if(get_dev()>0) oss<<" device="<<get_dev()<<",";
      oss<<"\b>";
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const SO3part& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 

