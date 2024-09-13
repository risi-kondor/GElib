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
#include "TensorUtils.hpp"
#include "BatchedTensor.hpp"


namespace GElib{


  template<typename GPART, typename TYPE>
  class Gpart: //public cnine::TensorView<TYPE>, 
    public cnine::BatchedTensor<TYPE>{
  public:

    using TENSOR=cnine::TensorView<TYPE>;
    using BASE=cnine::BatchedTensor<TYPE>;

    using Gdims=cnine::Gdims;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;

    using BASE::getb;

    using BASE::dim;
    using BASE::ndims;
    using BASE::device;
    using BASE::slice;
    using BASE::slices;
    using BASE::fuse_chunk;


    ~Gpart(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    Gpart(const Gdims& _dims, const int _fcode=0, const int _dev=0):
      BASE(_dims,_fcode,_dev){}

    Gpart(const int _b, const int _d, const int _nc, const int _fcode=0, const int _dev=0):
      BASE({_b,_d,_nc},_fcode,_dev){}
      
    Gpart(const int _b, const Gdims& _gdims, const int _d, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(Gdims(_b,_gdims,{_d,_nc}),_fcode,_dev){}
      
    Gpart(const cnine::BatchedTensor<TYPE>& M):
      BASE(M){}

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


    int getn() const{
      return dims.last();
    }

    Gdims get_gdims() const{
      return dims.chunk(1,dims.size()-3);
    }

    
  public: // ---- Batches -----------------------------------------------------------------------------------

    /*
    bool is_batched() const{
      return dims[0]>1;
    }

    int getb() const{
      return dims[0];
    }
    */

    //TENSOR batch(const int b) const{
    //return slice(0,b);
    //}

    /*
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

    template<typename TYPE2>
    void for_each_batch_multi(const cnine::TensorView<TYPE2>& x,
      const std::function<void(const int, const TENSOR& r, const cnine::TensorView<TYPE2>& x)>& lambda) const{
      //auto& r=static_cast<const GPART&>(*this);
      
      if(getb()==1){
	int B=x.dim(0);
	for(int b=0; b<B; b++)
	  lambda(b,slice(0,0),x.slice(0,b));
	return;
      }

      if(x.dim(0)==1){
	cnine::MultiLoop(getb(),[&](const int b){
	    lambda(b,slice(0,b),x.slice(0,0));
	  });
      }

      cnine::MultiLoop(getb(),[&](const int b){
	  lambda(b,slice(0,b),x.slice(0,b));
	});
     }


    void for_each_batch_multi(const GPART& x, const GPART& y,
      const std::function<void(const int, const TENSOR& r, const TENSOR& x, const TENSOR& y)>& lambda) const{
      auto& r=static_cast<const GPART&>(*this);
      int B=r.getb();
      GELIB_ASSRT(x.getb()==B);
      GELIB_ASSRT(y.getb()==B);
      cnine::MultiLoop(B,[&](const int b){
	  lambda(b,r.slice(0,b),x.slice(0,b),y.slice(0,b));
	});
    }
    */


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_grid() const{
      return dims.size()>3;
    }

    Gdims gdims() const{
      return dims.chunk(1,dims.size()-3);
    }

    int getg() const{
      GELIB_ASSRT(ndims()==4);
      return dims[1];
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

    GPART fuse_grid() const{
      if(!is_grid()){
	GPART r(*this);
	r.dims=r.dims.insert(1,1);
	r.strides=r.strides.insert(1,0);
	return r;
      }
      return static_cast<const GPART&>(*this).like(fuse_chunk(1,ndims()-3));
    }

    template<typename TYPE2>
    void for_each_cell_multi(const cnine::TensorView<TYPE2>& x, 
      const std::function<void(const int, const int, const TENSOR& r, const cnine::TensorView<TYPE2>& x)>& lambda) const{
      auto& r=static_cast<const GPART&>(*this);

      if(r.ndims()==3 && x.ndims()==3){
	r.template for_each_batch_multi<TYPE2>(x,[&](const int b, const TENSOR& r, const cnine::TensorView<TYPE2>& x){
	    lambda(b,0,r,x);
	  });
	return;
      }

      GPART _r=r.fuse_grid();
      auto _x=GElib::canonicalize(x);
      int G=std::max(_r.dims[1],_x.dims[1]);
      GELIB_ASSRT(_r.dims[1]==G || _r.dims[1]==1);
      GELIB_ASSRT(_x.dims[1]==G || _x.dims[1]==1);
      int mr=(_r.dims[1]>1);
      int mx=(_x.dims[1]>1);
      _r.template for_each_batch_multi<TYPE2>(_x,[&](const int b, const TENSOR& r, const cnine::TensorView<TYPE2>& x){
	  for(int g=0; g<G; g++)
	    lambda(b,g,r.slice(0,mr*g),x.slice(0,mx*g));
	});
    }

    void for_each_cell_multi(const GPART& x, const GPART& y,
      const std::function<void(const int, const int, const TENSOR& r, const TENSOR& x, const TENSOR& y)>& lambda) const{
      auto& r=static_cast<const GPART&>(*this);

      if(r.ndims()==3 && x.ndims()==3 && y.ndims()==3){
	r.for_each_batch_multi(x,y,[&](const int b, const TENSOR& r, const TENSOR& x, const TENSOR& y){
	    lambda(b,0,r,x,y);
	  });
	return;
      }

      GPART _r=r.fuse_grid();
      GPART _x=x.fuse_grid();
      GPART _y=y.fuse_grid();
      int G=std::max(std::max(_x.dims[1],_y.dims[1]),_r.dims[1]);
      GELIB_ASSRT(_r.dims[1]==G || _r.dims[1]==1);
      GELIB_ASSRT(_x.dims[1]==G || _x.dims[1]==1);
      GELIB_ASSRT(_y.dims[1]==G || _y.dims[1]==1);
      int mr=(_r.dims[1]>1);
      int mx=(_x.dims[1]>1);
      int my=(_y.dims[1]>1);
      _r.for_each_batch_multi(_x,_y,[&](const int b, const TENSOR& __r, const TENSOR& __x, const TENSOR& __y){
	  for(int g=0; g<G; g++)
	    lambda(b,g,__r.slice(0,mr*g),__x.slice(0,mx*g),__y.slice(0,my*g));
	});
    }


  public: // ---- Promotions ---------------------------------------------------------------------------------


    void canonicalize_to_4d(){
      int d=dims.size();
      if(d==3){
	dims=cnine::Gdims({dims[0],1,dims[1],dims[2]});
	strides=cnine::GstridesB({strides[0],0,strides[1],strides[2]});
      }
      if(d>4){
	reset(fuse_chunk(1,d-3));
      }
    }

    void promote_batch_to(const int b){
      if(dims[0]==b) return;
      GELIB_ASSRT(dims[0]==1);
      dims[0]=b;
      strides[0]=0;
    }

    void promote_grid_to(const int g){
      GELIB_ASSRT(ndims()==4);
      if(dims[1]==g) return;
      GELIB_ASSRT(dims[1]==1);
      dims[1]=g;
      strides[1]=0;
    }

    //TENSOR tile_channels(const int n){
    //GELIB_ASSRT(ndims()==4);
    //TENSOR r(*this);
    //dims=Gdims({dims[0],dims[1],dims[2],dims[3]/n,n});
    //return r;
    //}



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


    template<typename GPART2> // dummy template to break circular dependency
    GPART DiagCGproduct(const GPART2& y, const typename GPART2::IRREP_IX& l) const{
      auto& x=static_cast<const GPART&>(*this);
      int m=GPART::GROUP::CGmultiplicity(x.getl(),y.getl(),l);
      GELIB_ASSRT(m>0);
      GELIB_ASSRT(x.getn()==y.getn())
      GPART R(x.dominant_batch(y),x.dominant_gdims(y),l,x.getn(),0,dev);
      R.add_DiagCGproduct(x,y);
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
    /*
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
    */
    /*
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
    
    pair<GPART,GPART> co_promote_fused_grid(const GPART& y) const{
      GELIB_ASSRT(dims.size()==4);
      GELIB_ASSRT(y.dims.size()==4);
      int xg=dims[1];
      int yg=y.dims[1];
      if(xg==yg) return make_pair(*this,y);
      if(xg==1){
	GPART r(*this);
	r.dims[1]=yg;
	r.strides[1]=0;
	return make_pair(r,y);
      }
      if(yg==1){
	GPART r(y);
	r.dims[1]=xg;
	r.strides[1]=0;
	return make_pair(*this,r);
      }
      throw std::invalid_argument("Gelib error: the fused grid dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return make_pair(*this,y);
    }
    
    pair<GPART,GPART> fuse_and_co_promote(const GPART& y) const{
      auto [x1,y1]=fuse_grid().co_promote_batch(y.fuse_grid());
      auto [x2,y2]=x1.co_promote_fused_grid(y1);
      return make_pair(x2,y2);
    }
    */
    /*
    GPART canonicalize() const{
      return fuse_grid();
    }

    pair<GPART,GPART> co_canonicalize(const GPART& _y) const{
      GPART x=fuse_grid();
      GPART y=_y.fuse_grid();
      co_promote_sub<0>(x,y);
      co_promote_sub<1>(x,y);
      return make_pair(x,y);
    }

    template<int i>
    static void co_promote_sub(GPART& x, GPART& y){
      int dx=x.dims[i];
      int dy=y.dims[i];
      if(dx==dy) return; 
      if(dx==1){
	x.dims[i]=dy;
	return;
      }
      if(dy==1){
	y.dims[i]=dx;
	return;
      }
      throw std::invalid_argument("Gelib error: dimensions "+to_string(i)+" of "+x.repr()+" and "+y.repr()+
	" cannot be reconciled.");
    }
    */

