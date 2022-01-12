
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partA
#define _SO3partA

#include "CtensorA.hpp"
#include "SO3_CGbank.hpp"
#include "SO3_SPHgen.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{

  class SO3partA;

#ifdef _WITH_CUDA
  void SO3partA_CGproduct_cu(SO3partA& r, const SO3partA& x, const SO3partA& y, const int offs, 
    const cudaStream_t& stream, const int mode=0);
#endif

  class SO3partA: public cnine::CtensorA{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;

    using CtensorA::CtensorA;


  public: // ---- Constructors ------------------------------------------------------------------------------


    SO3partA(const int _l, const int _n, const int _nbu=-1, const int _dev=0):
      CtensorA({2*_l+1,_n},_nbu,_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partA(const int _l, const int _n, const FILLTYPE& dummy, const int _dev=0):
      CtensorA({2*_l+1,_n},-1,dummy,_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partA(const int _l, const int _n, const FILLTYPE& dummy, const device& _dev):
      CtensorA({2*_l+1,_n},-1,dummy,_dev.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partA(const int _l, const int _n, const int _nbu, const FILLTYPE& dummy, const int _dev=0):
      CtensorA({2*_l+1,_n},_nbu,dummy,_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partA(const int _l, const int _n, const int _nbu, const FILLTYPE& dummy, const device& _dev):
      CtensorA({2*_l+1,_n},_nbu,dummy,_dev.id()){}


  public: // ---- Copying -----------------------------------------------------------------------------------

    
    SO3partA(const SO3partA& x): 
      CtensorA(x){
    }

    SO3partA(const SO3partA& x, const cnine::nowarn_flag& dummy): 
      CtensorA(x){
    }

    SO3partA(SO3partA&& x): 
      CtensorA(std::move(x)){
    }

    SO3partA& operator=(const SO3partA& x){
      CtensorA::operator=(x);
      return *this;
    }

    SO3partA& operator=(SO3partA&& x){
      CtensorA::operator=(std::move(x));
      return *this;
    }


  public: // ---- Variants -----------------------------------------------------------------------------------


  public: // ---- Conversions -------------------------------------------------------------------------------


    explicit SO3partA(const CtensorA& x):
      CtensorA(x){
      //cout<<"CtensorA -> SO3partA"<<endl;
    }
    
    explicit SO3partA(CtensorA&& x):
      CtensorA(std::move(x)){
      //cout<<getl()<<getn()<<endl;
      //cout<<"move CtensorA -> SO3partA"<<endl;
    }
    

  public: // ---- Access -------------------------------------------------------------------------------------


    int getl() const{
      return (dims(0)-1)/2;
    }

    int getn() const{
      return dims(1);
    }

    complex<float> operator()(const int i, const int m) const{
      return CtensorA::get_value(m,i);
    }

    complex<float> get(const int i, const int m) const{
      return CtensorA::get_value(m,i);
    }

    complex<float> get_value(const int i, const int m) const{
      return CtensorA::get_value(m,i);
    }

    complex<float> getb(const int b, const int i, const int m) const{
      return CtensorA::get_value(b,m,i);
    }

    void set(const int i, const int m, complex<float> x){
      CtensorA::set(m,i,x);
    }

    void set_value(const int i, const int m, complex<float> x){
      CtensorA::set(m,i,x);
    }

    void set(const int b, const int i, const int m, complex<float> x){
      CtensorA::set(b,m,i,x);
    }

    void inc(const int i, const int m, complex<float> x){
      CtensorA::inc(m,i,x);
    }

    void inc(const int b, const int i, const int m, complex<float> x){
      CtensorA::inc(b,m,i,x);
    }

    SO3partA chunk(const int i, const int n=1) const {
      return SO3partA(CtensorA::chunk(1,i,n));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------



  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    //void add_CGproduct_cu(const SO3partA& x, const SO3partA& y, int offs, const cudaStream_t& stream);

    void add_CGproduct(const SO3partA& x, const SO3partA& y, int offs=0){

      if(dev==1){
#ifdef _WITH_CUDA
	assert(x.dev==1);
	assert(y.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	SO3partA_CGproduct_cu(*this,x,y,offs,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
	return; 
      }
      
      const int l=getl(); 
      const int l1=x.getl(); 
      const int l2=y.getl(); 
      const int N1=x.getn();
      const int N2=y.getn();
      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      if(nbu==-1){
	for(int n1=0; n1<N1; n1++){
	  //cout<<n1<<endl;
	  for(int n2=0; n2<N2; n2++){
	    //cout<<"  "<<n1<<endl;
	    for(int m1=-l1; m1<=l1; m1++){
	      //cout<<"    "<<m1<<endl;
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		//cout<<"      "<<m2<<endl;
		//cout<<x(n1,m1+l1)<<y(n2,m2+l2)<<endl;
		inc(offs+n2,m1+m2+l,C(m1+l1,m2+l2)*x(n1,m1+l1)*y(n2,m2+l2));
	      }
	    }
	  }
	  offs+=N2;
	}
	return;
      }

      assert(x.nbu==nbu);
      assert(y.nbu==nbu);

      /*
      for(int b=0; b<nbu; b++){
	int _offs=offs;
	for(int n1=0; n1<N1; n1++){
	  for(int n2=0; n2<N2; n2++)
	    for(int m1=-l1; m1<=l1; m1++)
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
		incb(b,_offs+n2,m1+m2+l,C(m1+l1,m2+l2)*x.getb(b,n1,m1+l1)*y.getb(b,n2,m2+l2));
	  _offs+=N2;
	}
      }
      */

    }


    void add_CGproduct_back0(const SO3partA& g, const SO3partA& y, int offs=0){

      if(dev==1){
#ifdef _WITH_CUDA
	assert(g.dev==1);
	assert(y.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	//g.CGproduct_g1cu(*this,y,offs,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
	return; 
      }

      const int l=g.getl(); 
      const int l1=getl(); 
      const int l2=y.getl(); 
      const int N1=getn();
      const int N2=y.getn();
      const SO3_CGcoeffs<float>& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      if(nbu==-1){
	for(int n1=0; n1<N1; n1++){
	  for(int n2=0; n2<N2; n2++){
	    for(int m1=-l1; m1<=l1; m1++)
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		inc(n1,m1+l1,C(m1+l1,m2+l2)*std::conj(y(n2,m2+l2))*g(offs+n2,m1+m2+l));
	      }
	  }
	  offs+=N2;
	}
      return;
      }
            
    }


    void add_CGproduct_back1(const SO3partA& g, const SO3partA& x, int offs=0){

      if(dev==1){
#ifdef _WITH_CUDA
	assert(g.dev==1);
	assert(x.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	//g.CGproduct_g2cu(x,*this,offs,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
	return; 
      }

      const int l=g.getl();
      const int l1=x.getl(); 
      const int l2=getl(); 
      const int N1=x.getn();
      const int N2=getn();
      const SO3_CGcoeffs<float>& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      if(nbu==-1){
	for(int n1=0; n1<N1; n1++){
	  for(int n2=0; n2<N2; n2++)
	    for(int m1=-l1; m1<=l1; m1++)
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
		inc(n2,m2+l2,C(m1+l1,m2+l2)*std::conj(x(n1,m1+l1))*g(offs+n2,m1+m2+l));
	  offs+=N2;
	}
	return ;
      }

    }


    void add_DiagCGproduct(const SO3partA& x, const SO3partA& y, const int offs=0){

      if(dev==1){
#ifdef _WITH_CUDA
	assert(x.dev==1);
	assert(y.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	//DiagCGproduct_cu(x,y,offs,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
	return; 
      }
      
      const int l=getl(); 
      const int l1=x.getl(); 
      const int l2=y.getl(); 
      const int N1=x.getn();
      const int N2=y.getn();
      assert(N1==N2);
      assert(N1==getn());
      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      if(nbu==-1){
	for(int n=0; n<N1; n++){
	  for(int m1=-l1; m1<=l1; m1++)
	    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
	      inc(offs+n,m1+m2+l,C(m1+l1,m2+l2)*x(n,m1+l1)*y(n,m2+l2));
	}
	return;
      }

      assert(x.nbu==nbu);
      assert(y.nbu==nbu);

      /*
      for(int b=0; b<nbu; b++){
	int _offs=offs;
	for(int n1=0; n1<N1; n1++){
	  for(int n2=0; n2<N2; n2++)
	    for(int m1=-l1; m1<=l1; m1++)
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
		incb(b,_offs+n2,m1+m2+l,C(m1+l1,m2+l2)*x.getb(b,n1,m1+l1)*y.getb(b,n2,m2+l2));
	  _offs+=N2;
	}
      }
      */

    }


  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    void add_spharm(const float x, const float y, const float z){
      int l=getl();

      float length=sqrt(x*x+y*y+z*z); 
      float len2=sqrt(x*x+y*y);
      complex<float> cphi(x/len2,y/len2);

      cnine::Gtensor<float> P=SO3_sphGen(l,z/length);
      vector<complex<float> > phase(l+1);
      phase[0]=complex<float>(1.0,0);
      for(int m=1; m<=l; m++)
	phase[m]=cphi*phase[m-1];
      
      for(int m=0; m<=l; m++){
	complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
	complex<float> b=complex<float>(1-2*(m%2))*std::conj(a);
	for(int j=0; j<getn(); j++){
	  inc(j,l+m,a);
	  inc(j,l-m,b);
	}
      }

    }


  public: // ---- rotations ----------------------------------------------------------------------------------


    SO3partA rotate(const SO3element& r){
      CtensorA D(WignerMatrix<float>(getl(),r),dev);
      SO3partA R(getl(),getn(),cnine::fill::zero,dev);
      R.add_Mprod_AA<0>(D,*this);
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SO3partA";
    }

    string describe() const{
      //if(!bundle) return "SO3partA"+dims.str()+"["+to_string(nbu)+"]";
      return "SO3partA"+dims.str();
    }

    string str(const string indent="") const{
      return gtensor().str(indent); //_transp(indent);
    }

   
  };


  /*
  inline SO3partA& asSO3partA(Cobject* x, const char* s){
    return downcast<SO3partA>(x,s);
  }

  inline SO3partA& asSO3partA(Cnode* x, const char* s){
    return downcast<SO3partA>(x,s);
  }
  
  inline SO3partA& asSO3partA(Cnode& x, const char* s){
    return downcast<SO3partA>(x,s);
  }
  */


  //#define SO3PARTB(x) asSO3partA(x,__PRETTY_FUNCTION__) 



}

#endif

  /*
  inline SO3partA& asSO3partB(Cobject* x){
    assert(x); 
    if(!dynamic_cast<SO3partB*>(x))
      cerr<<"GElib error: Cobject is of type "<<x->classname()<<" instead of SO3partB."<<endl;
    assert(dynamic_cast<SO3partB*>(x));
    return static_cast<SO3partB&>(*x);
  }

  inline SO3partB& asSO3partB(Cnode* x){
    assert(x->obj);
    if(!dynamic_cast<SO3partB*>(x->obj))
      cerr<<"GElib error: Cobject is of type "<<x->obj->classname()<<" instead of SO3partB."<<endl;
    assert(dynamic_cast<SO3partB*>(x->obj));
    return static_cast<SO3partB&>(*x->obj);
  }

  inline SO3partB& asSO3partB(Cnode& x){
    assert(x.obj);
    if(!dynamic_cast<SO3partB*>(x.obj))
      cerr<<"GElib error: Cobject is of type "<<x.obj->classname()<<" instead of SO3partB."<<endl;
    assert(dynamic_cast<SO3partB*>(x.obj));
    return static_cast<SO3partB&>(*x.obj);
  }
  */
    /*
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partA(const Gdims& _dims, const FILLTYPE& fill, const int _dev):
      cnine::CtensorA(_dims,fill,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partA(const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int _dev):
      cnine::CtensorA(_dims,_nbu,fill,_dev){}

    SO3partA(const Gdims& _dims, std::function<complex<float>(const int i, const int j)> fn):
      CtensorA(_dims,fn){}

    SO3partA(const Gtensor<complex<float> >& x, const int device=0):
      cnine::CtensorA(x,device){}
    */

    //public: // ---- Filled constructors -----------------------------------------------------------------------

    /*
    SO3partA(const SO3partA& x, const nowarn_flag& dummy): 
      CFtensor(x,dummy), dims(x.dims), nbu(x.nbu){}

    SO3partA* clone() const{
      return new SO3partA(*this, nowarn);
    }
    */

/*
#ifdef _WITH_CUDA
    void CGproduct_cu(const SO3partA& x, const SO3partA& y, int offs, const cudaStream_t& stream);
    void CGproduct_g1cu(SO3partA& xg, const SO3partA& y, int offs, const cudaStream_t& stream) const;
    void CGproduct_g2cu(const SO3partA& x, SO3partA& yg, int offs, const cudaStream_t& stream) const;
#endif 



*/
    /*
    SO3partA(const int l, const int n, const float x, const float y, const float z, const int nbu, const cnine::device& _dev=0):
      SO3partA({2*l+1,n},nbu,cnine::fill::raw,_dev.id()){

      float length=sqrt(x*x+y*y+z*z); 
      float len2=sqrt(x*x+y*y);
      complex<float> cphi(x/len2,y/len2);

      cnine::Gtensor<float> P=SO3_sphGen(l,z/length);
      vector<complex<float> > phase(l+1);
      phase[0]=complex<float>(1.0,0);
      for(int m=1; m<=l; m++)
	phase[m]=cphi*phase[m-1];
      
      for(int m=0; m<=l; m++){
	complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
	complex<float> b=complex<float>(1-2*(m%2))*std::conj(a);
	for(int j=0; j<n; j++){
	  set(j,l+m,a);
	  set(j,l-m,b);
	}
      }

    }
    */


    /* redundant 
    SO3partA(const SO3partA& x, const int dev): 
      CtensorA(x,dev){
    }
    */


