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
#include "TensorView.hpp"
#include "NamedTypes.hpp"
#include "MultiLoop.hpp"


namespace GElib{


  template<typename GPART, typename TYPE>
  class Gpart: public cnine::TensorView<TYPE>{
  public:

    typedef cnine::TensorView<TYPE> BASE;
    typedef cnine::TensorView<TYPE> TENSOR;

    typedef cnine::Gdims Gdims;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;

    using BASE::dim;
    using BASE::ndims;
    using BASE::device;
    using BASE::slice;
    using BASE::slices;

    //using BASE::bgfused_view3;

    // using BASE::is_batched;
    //using BASE::getb;
    //using BASE::nbatch;

    //using BASE::is_grid;
    //using BASE::gdims;
    //using BASE::cell;


    ~Gpart(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    Gpart(const Gdims& _dims, const int _fcode=0, const int _dev=0):
      BASE(_dims,_fcode,_dev){}

    Gpart(const int _b, const int _d, const int _nc, const int _fcode=0, const int _dev=0):
      BASE({_b,_d,_nc},_fcode,_dev){}
      
    Gpart(const int _b, const Gdims& _gdims, const int _d, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(Gdims(_b,_gdims,{_d,_nc}),_fcode,_dev){}
      
    void reset(const int _b, const Gdims& _gdims, const int _d, const int _nc, const int _fcode=0, const int _dev=0){
      BASE::reset(BASE(Gdims(_b,_gdims,{_d,_nc}),_fcode,_dev));
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


  public: // ---- Factory methods -------------------------------------------------------------------------------------


    Gpart zeros_like() const{
      return BASE::zeros_like();
    }

    Gpart zeros_like(const int d) const{
      return Gpart(dims.copy().set_back(1,d),0,dev);
    }

    Gpart zeros_like(const int d, const int n) const{
      return Gpart(dims.copy().set_back(1,d).set_back(0,n),0,dev);
    }


  public: // ---- Copying ---------------------------------------------------------------------------------------------

    

  public: // ---- Conversions ---------------------------------------------------------------------------------


    Gpart(const TENSOR& x):
      BASE(x){
      GELIB_ASSRT(ndims()>=3);
    }


  public: // ---- Access ----------------------------------------------------------------------------------------------


    bool is_grid() const{
      return dims.size()>3;
    }

    Gdims gdims() const{
      return dims.chunk(1,dims.size()-3);
    }

    int getn() const{
      return dims.last();
    }

    
  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return dims[0]>1;
    }

    int getb() const{
      return dims[0];
    }

    TENSOR batch(const int b) const{
      return slice(0,b);
    }

    void for_each_batch_multi(const GPART& x, const GPART& y,
      const std::function<void(const int, const TENSOR& r, const TENSOR& x, const TENSOR& y)>& lambda) const{
      auto& r=static_cast<const GPART&>(*this);
      int B=r.getb();
      GELIB_ASSRT(x.getb()==B);
      GELIB_ASSRT(y.getb()==B);
      cnine::MultiLoop(B,[&](const int b){
	  lambda(b,r.batch(b),x.batch(b),y.batch(b));
	});
    }


  public: // ---- Promotions ---------------------------------------------------------------------------------


    int dominant_batch(const GPART& y) const{
      int xb=getb();
      int yb=y.getb();
      if(xb==yb) return xb;
      if(xb==1) return yb;
      if(yb==1) return xb;
      throw std::invalid_argument("Gelib error: the batch dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return 0;
    }

    Gdims dominant_gdims(const GPART& y) const{
      Gdims xg=gdims();
      Gdims yg=y.gdims();
      if(xg==yg) return xg;
      if(!is_grid()) return yg;
      if(!y.is_grid()) return xg;
      throw std::invalid_argument("Gelib error: the grid dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return Gdims();
    }

    pair<GPART,GPART> co_promote_batch(const GPART& y) const{
      if(getb()==y.getb()) return make_pair(*this,y);
      int xb=getb();
      int yb=y.getb();
      if(xb==1){
	GPART r(*this);
	r.dims[0]=yb;
	r.strides[0]=0;
	return make_pair(r,y);
      }
      if(yb==1){
	GPART r(y);
	r.dims[0]=xb;
	r.strides[0]=0;
	return make_pair(*this,r);
      }
      throw std::invalid_argument("Gelib error: the batch dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return make_pair(*this,y);
    }
    
    pair<GPART,GPART> co_promote_grid(const GPART& y) const{
      if(!is_grid() && !y.is_grid()) return make_pair(*this,y);
      Gdims xg=gdims();
      Gdims yg=y.gdims();
      if(xg==yg) return make_pair(*this,y);
      if(xg.size()==0){
	//GPART r(arr,dims.insert(1,yg),strides.insert(1,yg.size(),0));
	GPART r(*this);
	r.dims=r.dims.insert(1,yg);
	r.strides.insert(1,yg.size(),0);
	return make_pair(r,y);
      }
      if(yg.size()==0){
	//GPART r(y.arr,y.dims.insert(1,xg),y.strides.insert(1,xg.size(),0));
	GPART r(y);
	r.dims=r.dims.insert(1,xg);
	r.strides.insert(1,xg.size(),0);
	return make_pair(*this,r);
      }
      throw std::invalid_argument("Gelib error: the grid dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return make_pair(*this,y);
    }
    
    pair<GPART,GPART> co_promote(const GPART& y) const{
      auto [x1,y1]=co_promote_batch(y);
      auto [x2,y2]=x1.co_promote_grid(y1);
      return make_pair(x2,y2);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


 
  public: // ---- CG-products --------------------------------------------------------------------------------


    template<typename GPART2> // dummy template to break circular dependency
    GPART CGproduct(const GPART2& y, const typename GPART2::IRREP_IX& l) const{
      auto& x=static_cast<const GPART&>(*this);
      int m=GPART::GROUP::CGmultiplicity(x.getl(),y.getl(),l);
      GELIB_ASSRT(m>0);
      GPART R(x.dominant_batch(y),x.dominant_gdims(y),l,x.getn()*y.getn(),0,dev);
      R.add_CGproduct(x,y);
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::Gpart";
    }

    string repr() const{
      ostringstream oss;
      return oss.str();
    }
    
    string str(const string indent="") const{
      if(getb()==1) return slice(0,0).str(indent);
      ostringstream oss;
      for(int i=0; i<getb(); i++){
	oss<<indent<<"Batch "<<i<<":"<<endl;
	oss<<slice(0,i).str(indent+"  ");
      }
      return oss.str();
    }

    string to_print(const string indent="") const{
      ostringstream oss;
      oss<<indent<<dynamic_cast<const GPART&>(*this).repr()<<":"<<endl;
      oss<<str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gpart& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 

  /*
  template<typename OBJ>
  int dominant_batch(const OBJ& x, const OBJ& y){
    int xb=x.getb();
    int yb=y.getb();
    if(xb==yb) return xb;
    if(xb==1) return yb;
    if(yb==1) return xb;
    throw std::invalid_argument("Gelib error: the batch dimensions of two objects can only be reconciled if they are the same or one of them is 1.");
    return 0;
  }

  template<typename OBJ>
  Gdims dominant_gdims(const OBJ& x, const OBJ& y){
    Gdims xg=x.gdims();
    Gdims yg=y.gdims();
    if(xg==yg) return xg;
    if(!x.is_grid()) return yg;
    if(!y.is_grid()) return xg;
    throw std::invalid_argument("Gelib error: the grid dimensions of two objects can only be reconciled if they are the same or one of them is none.");
    return Gdims();
  }

  template<typename GPART, typename TYPE>
  inline std::pair<Gpart<GPART,TYPE>,Gpart<GPART,TYPE> >
  co_promote(const Gpart<GPART,TYPE>& x, const Gpart<GPART,TYPE>& y){

    if(x.getb()==y.getb() && !x.is_grid() && !y.is_grid()) 
      return make_pair(x,y);

    int xb=x.getb();
    int yb=y.getb();
    

  }

  template<typename GPART, typename TYPE>
  inline std::pair<Gpart<GPART,TYPE>,Gpart<GPART,TYPE> >
  co_promote(const Gpart<GPART,TYPE>& x, const Gpart<GPART,TYPE>& y){

    if(x.getb()==y.getb() && !x.is_grid() && !y.is_grid()) 
      return make_pair(x,y);

    int xb=x.getb();
    int yb=y.getb();
    

  }
  */
