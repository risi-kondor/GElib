// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB
#define _SO3partB

#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
//#include "FakeGrad.hpp"
//#include "SO3Fpart3_view.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"

#include "SO3part_addCGsquareFn.hpp"

#include "SO3part_addFproduct_Fn.hpp"
//#include "SO3part_addFproductB_Fn.hpp"
#include "SO3part_addFproduct_back0Fn.hpp"
//#include "SO3part_addFproductB_back0Fn.hpp"
#include "SO3part_addFproduct_back1Fn.hpp"

//#include "SO3_CGbank.hpp"
#include "RtensorA.hpp"
#include "SO3_SPHgen.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{


  // An SO3partB is a  b x (2l+1) x n   dimensional complex tensor.


  class SO3partB: public cnine::CtensorB 
  {
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    
    using CtensorB::CtensorB;

#ifdef WITH_FAKE_GRAD
    SO3partB* grad=nullptr;
#endif 

    ~SO3partB(){
#ifdef WITH_FAKE_GRAD
      if(!is_view) delete grad;
#endif 
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3partB(const int b, const int l, const int n, const int _dev=0):
      CtensorB(Gdims(b,2*l+1,n),_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partB(const int b, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorB(Gdims(b,2*l+1,n),dummy,_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3partB raw(const int l, const int n){
      return SO3partB(1,l,n,cnine::fill_raw());}
    static SO3partB zero(const int l, const int n){
      return SO3partB(1,l,n,cnine::fill_zero());}
    static SO3partB gaussian(const int l, const int n){
      return SO3partB(1,l,n,cnine::fill_gaussian());}

    static SO3partB raw(const int b, const int l, const int n,  const int _dev=0){
      return SO3partB(b,l,n,cnine::fill_raw());}
    static SO3partB zero(const int b, const int l, const int n,  const int _dev=0){
      return SO3partB(b,l,n,cnine::fill_zero(),_dev);}
    static SO3partB gaussian(const int b, const int l, const int n,  const int _dev=0){
      return SO3partB(b,l,n,cnine::fill_gaussian(),_dev);}


    static SO3partB Fraw(const int b, const int l, const int _dev=0){
      return SO3partB(1,l,2*l+1,cnine::fill_raw());}
    static SO3partB Fzero(const int b, const int l, const int _dev=0){
      return SO3partB(b,l,2*l+1,cnine::fill_zero(),_dev);}
    static SO3partB Fgaussian(const int b, const int l, const int _dev=0){
      return SO3partB(b,l,2*l+1,cnine::fill_gaussian(),_dev);}


  public: // ---- Copying ------------------------------------------------------------------------------------
    // only needed for grad

    #ifdef WITH_FAKE_GRAD
    SO3partB(const SO3partB& x):
      CtensorB(x){
      if(x.grad) grad=new SO3partB(*x.grad);
    }
      
    SO3partB(SO3partB&& x):
      CtensorB(std::move(x)){
      grad=x.grad;
      x.grad=nullptr;
    }
    #endif 

  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partB(const CtensorB& x):
      CtensorB(x){
      assert(dims.size()==3);
      assert(dims(1)%2==1);
    }
      
    SO3partB(CtensorB&& x):
      CtensorB(std::move(x)){
      assert(dims.size()==3);
      assert(dims(1)%2==1);
    }
      

  public: // ---- Transport -----------------------------------------------------------------------------------


    SO3partB to_device(const int _dev) const{
      return SO3partB(*this,_dev);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getb() const{
      return dims(0);
    }
    
    int getl() const{
      return (dims(1)-1)/2;
    }

    int getn() const{
      return dims(2);
    }

    bool is_F() const{
      return (dims(1)==dims(2));
    }

        
  public: // ---- Access views --------------------------------------------------------------------------------


    SO3partB _view(){
      return CtensorB::view();
    }

    SO3part3_view view() const{
      if(dev==0) return SO3part3_view(arr,dims,strides,coffs);
      else return SO3part3_view(arrg,dims,strides,coffs,dev);
    }

    operator SO3part3_view() const{
      if(dev==0) return SO3part3_view(arr,dims,strides,coffs);
      else return SO3part3_view(arrg,dims,strides,coffs,dev);
    }


    //SO3Fpart3_view Fview() const{
    //if(dev==0) return SO3Fpart3_view(arr,dims,strides,coffs);
    //else return SO3Fpart3_view(arrg,dims,strides,coffs,dev);
    //}


  public: // ---- Operations ---------------------------------------------------------------------------------


    SO3partB operator*(const CtensorB& M) const{
      assert(M.dims.size()==2);
      assert(M.dims(0)==getn());
      SO3partB R(getb(),getl(),M.dims(1),cnine::fill_zero());
      R.add_mprod(*this,M);
      return R;
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add_mprod(const SO3partB& x, const CtensorB& M){
      const int B=getb();
      assert(x.getb()==B);
      auto view=view3();
      auto xview=x.view3();
      auto Mview=M.view2();
      cnine::MultiLoop(B,[&](const int b){view.slice0(b).add_matmul_AA(xview.slice0(b),Mview);});
    }


  public: // ---- Rotations ----------------------------------------------------------------------------------


    SO3partB rotate(const SO3element& r){
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      SO3partB R=SO3partB::zero(getb(),getl(),getn(),dev);

      int B=getb();
      auto dv=D.view2D();
      auto xv=this->view3D();
      auto rv=R.view3D();
      
      for(int b=0; b<B; b++){
	auto v=rv.slice0(b);
	cnine::Ctensor_add_mprod_AA()(v,dv,xv.slice0(b));
      }

      return R;
    }


  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    static SO3partB spharm(const int b, const int l, const int n, const float x, const float y, const float z, const int _dev=0){
      SO3partB R=SO3partB(b,l,n,cnine::fill_raw(),_dev);
      R.add_spharm(x,y,z);
      if(_dev>0) return SO3partB(R,_dev);
      return R;
    }

    void add_spharm(const float x, const float y, const float z){
      int l=getl();
      int B=getb();
      int n=getn();
      cnine::Ctensor3_view v=view3();

      float length=sqrt(x*x+y*y+z*z); 
      float len2=sqrt(x*x+y*y);
      if(len2==0 || std::isnan(x/len2) || std::isnan(y/len2)){
	float a=sqrt(((float)(2*l+1))/(M_PI*4.0));
	for(int i=0; i<B; i++)
	  for(int j=0; j<n; j++)
	    v.inc(i,l,j,a);
	return;
      }
      complex<float> cphi(x/len2,y/len2);

      cnine::Gtensor<float> P=SO3_sphGen(l,z/length);
      vector<complex<float> > phase(l+1);
      phase[0]=complex<float>(1.0,0);
      for(int m=1; m<=l; m++)
	phase[m]=cphi*phase[m-1];
      
      for(int m=0; m<=l; m++){
	complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
	complex<float> b=complex<float>(1-2*(m%2))*std::conj(a);
	for(int i=0; i<B; i++){
	  for(int j=0; j<n; j++){
	    v.inc(i,l+m,j,a);
	    if(m>0) v.inc(i,l-m,j,b);
	  }
	}
      }
    }


    void add_spharm(const cnine::RtensorA& x){
      int l=getl();
      int B=getb();
      int n=getn();
      cnine::Ctensor3_view v=view3();
      assert(x.dims.size()==3);
      assert(x.dims[0]==B);
      assert(x.dims[1]==3);
      assert(x.dims[2]==n);

      for(int b=0; b<B; b++){
	for(int j=0; j<n; j++){
	  float vx=x(b,0,j);
	  float vy=x(b,1,j);
	  float vz=x(b,2,j);
	  float length=sqrt(vx*vx+vy*vy+vz*vz); 
	  float len2=sqrt(vx*vx+vy*vy);

	  if(len2==0 || std::isnan(vx/len2) || std::isnan(vy/len2)){
	    float a=sqrt(((float)(2*l+1))/(M_PI*4.0));
	    for(int j=0; j<n; j++)
	      v.inc(b,l,j,a);
	    return;
	  }

	  complex<float> cphi(vx/len2,vy/len2);
	  cnine::Gtensor<float> P=SO3_sphGen(l,vz/length);
	  vector<complex<float> > phase(l+1);
	  phase[0]=complex<float>(1.0,0);
	  for(int m=1; m<=l; m++)
	    phase[m]=cphi*phase[m-1];
      
	  for(int m=0; m<=l; m++){
	    complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
	    complex<float> aa=complex<float>(1-2*(m%2))*std::conj(a);
	    v.inc(b,l+m,j,a);
	    if(m>0) v.inc(b,l-m,j,aa);
	  }

	}
      }
    }


    void add_spharmB(const cnine::RtensorA& x){ // deprecated
      int l=getl();
      int B=getb();
      int n=getn();
      cnine::Ctensor3_view v=view3();
      assert(x.dims.size()==2);
      assert(x.dims[0]==B);
      assert(x.dims[1]==3);

      for(int b=0; b<B; b++){
	float vx=x(b,0);
	float vy=x(b,1);
	float vz=x(b,2);
	float length=sqrt(vx*vx+vy*vy+vz*vz); 
	float len2=sqrt(vx*vx+vy*vy);
	if(len2==0 || std::isnan(vx/len2) || std::isnan(vy/len2)){
	  float a=sqrt(((float)(2*l+1))/(M_PI*4.0));
	  for(int j=0; j<n; j++)
	    v.inc(b,l,j,a);
	  return;
	}
	complex<float> cphi(vx/len2,vy/len2);
	cnine::Gtensor<float> P=SO3_sphGen(l,vz/length);
	vector<complex<float> > phase(l+1);
	phase[0]=complex<float>(1.0,0);
	for(int m=1; m<=l; m++)
	  phase[m]=cphi*phase[m-1];
      
	for(int m=0; m<=l; m++){
	  complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
	  complex<float> aa=complex<float>(1-2*(m%2))*std::conj(a);
	  for(int j=0; j<n; j++){
	    v.inc(b,l+m,j,a);
	    if(m>0) v.inc(b,l-m,j,aa);
	  }
	}
      }
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    SO3partB CGproduct(const SO3partB& y, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      SO3partB R=SO3partB::zero(getb(),l,getn()*y.getn(),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }

    void add_CGproduct(const SO3partB& x, const SO3partB& y, const int _offs=0){
      SO3part_addCGproductFn()(*this,x,y,_offs);
    }

    void add_CGproduct_back0(const SO3partB& g, const SO3partB& y, const int _offs=0){
      SO3part_addCGproduct_back0Fn()(*this,g,y,_offs);
    }

    void add_CGproduct_back1(const SO3partB& g, const SO3partB& x, const int _offs=0){
      SO3part_addCGproduct_back1Fn()(*this,g,x,_offs);
    }


    // ---- BlockedCGproduct 


    SO3partB BlockedCGproduct(const SO3partB& y, const int bsize, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      assert(getn()==y.getn());
      SO3partB R=SO3partB::zero(getb(),l,getn()*bsize,get_dev());
      R.add_BlockedCGproduct(*this,y,bsize);
      return R;
    }

    void add_BlockedCGproduct(const SO3partB& x, const SO3partB& y, const int bsize, const int _offs=0){
      SO3part_addBlockedCGproductFn()(*this,x,y,bsize,_offs);
    }

    void add_BlockedCGproduct_back0(const SO3partB& g, const SO3partB& y, const int bsize, const int _offs=0){
      SO3part_addBlockedCGproduct_back0Fn()(*this,g,y,bsize,_offs);
    }

    void add_BlockedCGproduct_back1(const SO3partB& g, const SO3partB& x, const int bsize, const int _offs=0){
      SO3part_addBlockedCGproduct_back1Fn()(*this,g,x,bsize,_offs);
    }


    // ---- DiagCGproduct 


    SO3partB DiagCGproduct(const SO3partB& y, const int l) const{
      return BlockedCGproduct(y,1,l);
    }

    void add_DiagCGproduct(const SO3partB& x, const SO3partB& y, const int _offs=0){
      add_BlockedCGproduct(x,y,1,_offs);
    }

    void add_DiagCGproduct_back0(const SO3partB& g, const SO3partB& y, const int _offs=0){
      add_BlockedCGproduct_back0(g,y,1,_offs);
    }

    void add_DiagCGproduct_back1(const SO3partB& g, const SO3partB& x, const int _offs=0){
      add_BlockedCGproduct_back1(g,x,1,_offs);
    }


    // ---- CGsquare


    SO3partB CGsquare(const int l) const{
      assert(l>=0 && l<=2*getl());
      int parity=(2*getl()-l)%2;
      SO3partB R=SO3partB::zero(getb(),l,getn()*(getn()+1-2*parity)/2,get_dev());
      R.add_CGsquare(*this);
      return R;
    }

    void add_CGsquare(const SO3partB& x, const int _offs=0){
      SO3part_addCGsquareFn()(*this,x,_offs);
    }



  public: // ---- F-products --------------------------------------------------------------------------------


    void add_Fproduct(const SO3partB& x, const SO3partB& y){
      SO3part_addFproduct_Fn()(view3(),x.view3(),y.view3());
    }

    void add_FproductB(const SO3partB& x, const SO3partB& y){
      SO3part_addFproduct_Fn(0,1)(view3(),x.view3(),y.view3());
    }

    void add_Fproduct_back0(const SO3partB& g, const SO3partB& y){
      SO3part_addFproduct_back0Fn()(view3(),g.view3(),y.view3());
    }

    void add_FproductB_back0(const SO3partB& g, const SO3partB& y){
      SO3part_addFproduct_back0Fn(0,1)(view3(),g.view3(),y.view3());
    }

    void add_Fproduct_back1(const SO3partB& g, const SO3partB& x){
      SO3part_addFproduct_back1Fn()(view3(),g.view3(),x.view3());
    }

    void add_FproductB_back1(const SO3partB& g, const SO3partB& x){
      SO3part_addFproduct_back1Fn(0,1)(view3(),g.view3(),x.view3());
    }


  public: // ---- Experimental -------------------------------------------------------------------------------



    #ifdef WITH_FAKE_GRAD
    void add_to_grad(const SO3partB& x){
      if(grad) grad->add(x);
      else grad=new SO3partB(x);
    }

    SO3partB& get_grad(){
      if(!grad) grad=new SO3partB(SO3partB::zeros_like(*this));
      return *grad;
    }

    SO3partB view_of_grad(){
      if(!grad) grad=new SO3partB(SO3partB::zeros_like(*this));
      return SO3partB(grad->CtensorB::view());
    }
    #endif 


  public: // ---- I/O ----------------------------------------------------------------------------------------

    string repr(const string indent="") const{
      return "<GElib::SO3partB(l="+to_string(getl())+",n="+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3partB& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
