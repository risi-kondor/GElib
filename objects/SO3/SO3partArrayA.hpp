
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partArrayA
#define _SO3partArrayA

#include "CtensorArrayA.hpp"
#include "SO3part.hpp"
#include "SO3_CGbank.hpp"
#include "SO3_SPHgen.hpp"
//#include "Cmaps.hpp"
#include "CellwiseBinaryCmap.hpp"

//#include "cell_ops/SO3partA_CGproduct_op.hpp"
#include "SO3partA_CGproduct_cop.hpp"
#include "SO3partA_CGproduct_back0_cop.hpp"
#include "SO3partA_CGproduct_back1_cop.hpp"

#include "SO3partA_DiagCGproduct_cop.hpp"


extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{


  class SO3partArrayA: public cnine::CtensorArrayA{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    int l;
    int n;


  public: // ---- Constructors ------------------------------------------------------------------------------


    SO3partArrayA(const Gdims& _adims, const int _l, const int _n, const int _nbu=-1, const int _dev=0):
      CtensorArrayA(_adims,{2*_l+1,_n},_nbu,_dev), l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partArrayA(const Gdims& _adims, const int _l, const int _n, const FILLTYPE& dummy, const int _dev=0):
      CtensorArrayA(_adims, {2*_l+1,_n},-1,dummy,_dev), l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partArrayA(const Gdims& _adims, const int _l, const int _n, const FILLTYPE& dummy, const device& _dev):
      CtensorArrayA(_adims, {2*_l+1,_n},-1,dummy,_dev.id()), l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partArrayA(const Gdims& _adims, const int _l, const int _n, const int _nbu, const FILLTYPE& dummy, const int _dev=0):
      CtensorArrayA(_adims, {2*_l+1,_n},_nbu,dummy,_dev), l(_l), n(_n){}

    SO3partArrayA(const Gdims& _adims, const int _l, const int _n, const int _nbu, const cnine::fill_view& dummy, 
      float* _arr, float* _arrc, const int _dev=0): 
      CtensorArrayA(_adims,{2*_l+1,_n},_nbu,dummy,_arr,_arrc,_dev){}
  
    SO3partArrayA(const Gdims& _adims, const int _l, const int _n, const cnine::fill_view& dummy, 
      float* _arr, float* _arrc, const int _dev=0): 
      SO3partArrayA(_adims,_l,_n,-1,dummy,_arr,_arrc,_dev){}
  

  public: // ---- Copying -----------------------------------------------------------------------------------

    
    SO3partArrayA(const SO3partArrayA& x): 
      CtensorArrayA(x), l(x.l), n(x.n){
    }

    SO3partArrayA(const SO3partArrayA& x, const cnine::nowarn_flag& dummy): 
      CtensorArrayA(x), l(x.l), n(x.n){
    }

    SO3partArrayA(SO3partArrayA&& x): 
      CtensorArrayA(std::move(x)), l(x.l), n(x.n){
    }

    SO3partArrayA& operator=(const SO3partArrayA& x){
      CtensorArrayA::operator=(x);
      l=x.l; n=x.n;
      return *this;
    }

    SO3partArrayA& operator=(SO3partArrayA&& x){
      CtensorArrayA::operator=(std::move(x));
      l=x.l; n=x.n;
      return *this;
    }


    SO3partArrayA(const SO3partArrayA& x, const int dev): 
      CtensorArrayA(x,dev), l(x.l), n(x.n){
    }

    SO3partArrayA(const SO3partArrayA& x, const cnine::view_flag& flag): 
      CtensorArrayA(x,flag), l(x.l), n(x.n){
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    SO3partArrayA(const CtensorArrayA& x):
      CtensorArrayA(x){
      assert(x.cdims.size()==2 || x.cdims.size()==3);
      l=(x.cdims[0]-1)/2;
      n=x.cdims[1];
    }
    
    SO3partArrayA(CtensorArrayA&& x):
      CtensorArrayA(std::move(x)){
      assert(x.cdims.size()==2 || x.cdims.size()==3);
      l=(x.cdims[0]-1)/2;
      n=x.cdims[1];
    }

    //SO3partArrayA(const SO3partA& x):
    //CtensorArrayA(x)

  public: // ---- Transport ----------------------------------------------------------------------------------


    SO3partArrayA& move_to(const device& _dev){
      CtensorArrayA::move_to_device(_dev.id());
      return *this;
    }
    
    SO3partArrayA& move_to_device(const int _dev){
      CtensorArrayA::move_to_device(_dev);
      return *this;
    }
    
    SO3partArrayA to(const device& _dev) const{
      return SO3partArrayA(CtensorArrayA(*this,_dev.id()));
    }

    SO3partArrayA to_device(const int _dev) const{
      return SO3partArrayA(CtensorArrayA(*this,_dev));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getl() const{
      return l;
    }

    int getn() const{
      return n;
    }


    //SO3part_spec cspec() const{
    //return SO3part_spec(l,n,dims,nbu,cstrides,asize);
    //}


  public: // ---- Cell Access --------------------------------------------------------------------------------


    SO3partA get_cell(const Gindex& aix) const{
      return SO3partA(CtensorArrayA::get_cell(aix));
    }

    void copy_cell_into(SO3partA& R, const Gindex& aix) const{
      CtensorArrayA::copy_cell_into(R,aix);
    }

    void add_cell_into(SO3partA& R, const Gindex& aix) const{
      CtensorArrayA::copy_cell_into(R,aix);
    }

    //void set_cell(const Gindex& aix, const SO3partA& x) const{
    //CtensorArrayA::set_cell(aix,x);
    //}

    //void add_to_cell(const Gindex& aix, const SO3partA& x) const{
    //CtensorArrayA::set_cell(aix,x);
    //}

    
    const SO3partA cell(const Gindex& aix) const{
      return SO3partA(CtensorArrayA::cell(aix));
    }

    const SO3partA cell(const int ix) const{
      return SO3partA(CtensorArrayA::cell(ix));
    }

    SO3partA cell(const Gindex& aix){
      return SO3partA(CtensorArrayA::cell(aix));
    }

    SO3partA cell(const int ix){
      return SO3partA(CtensorArrayA::cell(ix));
    }

    //SO3partA* cellp(const Gindex& aix){
    //return new SO3partA(CtensorArrayA::cell(aix));
    //}


  public: // ---- Operations ---------------------------------------------------------------------------------



  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add_CGproduct(const SO3partArrayA& x, const SO3partArrayA& y, const int offs){
      cnine::add_cellwise<SO3partA_CGproduct_cop>(*this,x,y,offs);
      //SO3partA_add_CGproduct_cop op(offs);
      //cnine::CellwiseBiCmap map(op,*this,x,y);
    }

    void add_CGproduct_back0(const SO3partArrayA& g, const SO3partArrayA& y, const int offs){
      cnine::add_cellwise<SO3partA_CGproduct_back0_cop>(*this,g,y,offs);
    }

    void add_CGproduct_back1(const SO3partArrayA& x, const SO3partArrayA& g, const int offs){
      cnine::add_cellwise<SO3partA_CGproduct_back1_cop>(*this,x,g,offs);
    }

    /*
    void add_inner_CGproduct(const SO3partArrayA& x, const SO3partArrayA& y, const int offs){
      SO3partA_add_CGproduct_cop op(offs);
      cnine::InnerBiCmap map(op,*this,x,y);
    }

    void add_outer_CGproduct(const SO3partArrayA& x, const SO3partArrayA& y, const int offs){
      SO3partA_add_CGproduct_cop op(offs);
      cnine::OuterBiCmap map(op,*this,x,y);
    }

    void add_convolve_CGproduct(const SO3partArrayA& x, const SO3partArrayA& y, const int offs){
      SO3partA_add_CGproduct_cop op(offs);
      if(x.ak==1) cnine::Convolve1BiCmap map(op,*this,x,y);
      if(x.ak==2) cnine::Convolve2BiCmap map(op,*this,x,y);
    }

    void add_CGproduct(const SO3partA& x, const SO3partArrayA& y, const int offs){
      SO3partA_add_CGproduct_cop op(offs);
      cnine::BroadcastLeftBiCmap map(op,*this,SO3partArrayA(x),y);
    }

    void add_CGproduct(const SO3partArrayA& x, const SO3partA& y, const int offs){
      SO3partA_add_CGproduct_cop op(offs);
      cnine::BroadcastRightBiCmap map(op,*this,x,SO3partArrayA(y));
    }


    void add_CGproduct_back0(const SO3partArrayA& g, const SO3partArrayA& y, const int offs){
      SO3partA_add_CGproduct_back0_cop op(offs);
      cnine::CellwiseBiCmap map(op,*this,g,y);
    }

    void add_CGproduct_back1(const SO3partArrayA& g, const SO3partArrayA& x, const int offs){
      SO3partA_add_CGproduct_back1_cop op(offs);
      cnine::CellwiseBiCmap map(op,*this,g,x);
    }
    */


  public: // ---- Broadcast cumulative operations ------------------------------------------------------------

    
    void broadcast_add_mprod(const cnine::CtensorA& x, const cnine::CtensorArrayA& y){ // TODO
      broadcast_add_Mprod_AA<0>(x,y);
    }

    //void broadcast_add_mprod(const cnine::CtensorArrayA& x, const cnine::CtensorA& y){
    //broadcast_add_Mprod_AA<0>(x,y);
    //}
    

  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    /*
    SO3partArrayA(const int l, const int n, const float x, const float y, const float z, const int nbu, const cnine::device& _dev=0):
      SO3partArrayA({2*l+1,n},nbu,cnine::fill::raw,_dev.id()){

      float length=sqrt(x*x+y*y+z*z); 
      float len2=sqrt(x*x+y*y);
      complex<float> cphi(x/len2,y/len2);

      Gtensor<float> P=SO3_sphGen(l,z/length);
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



  public: // ---- I/O ----------------------------------------------------------------------------------------

    string classname() const{
      return "SO3partArrayA";
    }

    string describe() const{
      return "SO3partArrayA"+dims.str();
    }

    string str(const string indent="") const{
      ostringstream oss;

      for(int i=0; i<aasize; i++){
	Gindex aix(i,adims);
	oss<<indent<<"Cell "<<aix<<endl;
	oss<<get_cell(aix).str(indent)<<endl<<endl;
      }

      return oss.str();

    }
    

   
  };

}

#endif 

/*

#ifdef _WITH_CUDA
    void CGproduct_cu(const SO3partArrayA& x, const SO3partArrayA& y, int offs, const cudaStream_t& stream);
    void CGproduct_g1cu(SO3partArrayA& xg, const SO3partArrayA& y, int offs, const cudaStream_t& stream) const;
    void CGproduct_g2cu(const SO3partArrayA& x, SO3partArrayA& yg, int offs, const cudaStream_t& stream) const;
#endif 


    void add_CGproduct(const SO3partArrayA& x, const SO3partArrayA& y, int offs=0){

      if(dev==1){
#ifdef _WITH_CUDA
	x.to_device(1);
	y.to_device(1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	CGproduct_cu(x,y,offs,stream);
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
	  for(int n2=0; n2<N2; n2++)
	    for(int m1=-l1; m1<=l1; m1++)
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		cout<<m1<<m2<<endl;
		inc(offs+n2,m1+m2+l,C(m1+l1,m2+l2)*x(n1,m1+l1)*y(n2,m2+l2));
	      }
	  offs+=N2;
	}
	return;
      }
      
      assert(x.nbu==nbu);
      assert(y.nbu==nbu);

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

    }


    void add_CGproduct_back0(const SO3partArrayA& g, const SO3partArrayA& y, int offs=0){

      if(dev==1){
#ifdef _WITH_CUDA
	g.to_device(1);
	y.to_device(1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	g.CGproduct_g1cu(*this,y,offs,stream);
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
      const SO3_CGcoeffs<float>& C=SO3_cgbank.get<float>(l1,l2,l);

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


    void add_CGproduct_back1(const SO3partArrayA& g, const SO3partArrayA& x, int offs=0){

      if(dev==1){
#ifdef _WITH_CUDA
	x.to_device(1);
	g.to_device(1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	g.CGproduct_g2cu(x,*this,offs,stream);
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
      const SO3_CGcoeffs<float>& C=SO3_cgbank.get<float>(l1,l2,l);

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

*/
