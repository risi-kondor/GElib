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

#ifndef _GElibGpart
#define _GElibGpart

#include "GElib_base.hpp"
#include "TensorView.hpp"
#include "NamedTypes.hpp"
#include "MultiLoop.hpp"
#include "TensorUtils.hpp"
#include "GatherSlices.hpp"
#include "BatchedTensor.hpp"


namespace GElib{


  template<typename GPART, typename TYPE>
  class Gpart: //public cnine::TensorView<TYPE>, 
    public cnine::BatchedTensor<TYPE>{
  public:

    using TENSOR=cnine::TensorView<TYPE>;
    using BASE=cnine::BatchedTensor<TYPE>;

    using Gdims=cnine::Gdims;
    using GstridesB=cnine::GstridesB;

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


    bool promote_batch_to(const int b){
      if(dims[0]==b) return true;
      if(dims[0]!=1) return false;
      dims[0]=b;
      strides[0]=0;
      return true;
    }

    bool reconcile_batches(Gpart& x, Gpart& y){
      int b=std::max(getb(),std::max(x.getb(),y.getb()));
      if(dims[0]!=1 &&  dims[0]!=b) return false;
      if(x.dims[0]!=1 &&  x.dims[0]!=b) return false;
      if(y.dims[0]!=1 &&  y.dims[0]!=b) return false;
      promote_batch_to(b);
      x.promote_batch_to(b);
      y.promote_batch_to(b);
      return true;
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_grid() const{
      return dims.size()>3;
    }

    bool grid_is_fusible() const{
      if(!is_grid()) return true;
      auto [s,d]=strides.chunk(1,ndims()-3).fuser(dims.chunk(1,ndims()-3));
      return d!=-1;
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


  public: // ---- Lambdas ------------------------------------------------------------------------------------


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

      if(r.grid_is_fusible() && x.grid_is_fusible() && y.grid_is_fusible()){
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
	return;
      }

      if((r.ndims()==3 || r.ndims()==5) && (x.ndims()==3 || x.ndims()==5) && (y.ndims()==3 || y.ndims()==5)){
	int g0=1;
	int g1=1;
	if(r.ndims()==5){g0=r.dims[1]; g1=r.dims[2];}
	if(x.ndims()==5){g0=x.dims[1]; g1=x.dims[2];}
	if(y.ndims()==5){g0=y.dims[1]; g1=y.dims[2];}
	GPART _r(r);
	GPART _x(x);
	GPART _y(y);
	_r.promote_grid_to(g0,g1);
	_x.promote_grid_to(g0,g1);
	_y.promote_grid_to(g0,g1);
	_r.for_each_batch_multi(_x,_y,[&](const int b, const TENSOR& __r, const TENSOR& __x, const TENSOR& __y){
	    for(int _g0=0; _g0<g0; _g0++)
	      for(int _g1=0; _g1<g1; _g1++)
		lambda(b,_g0*g1+_g1,__r.slice(0,_g0).slice(0,_g1),__x.slice(0,_g0).slice(0,_g1),__y.slice(0,_g0).slice(0,_g1));
	  });
	return;
      }
      
      GELIB_ERROR("Error: if the grid dimensions are not fusible then the grid dimension of each tensor must be 2.");
    }


    void for_each_cell_multi(const GPART& x, const GPART& y,
      const std::function<void(const TENSOR& r, const TENSOR& x, const TENSOR& y)>& lambda) const{
      auto& r=static_cast<const GPART&>(*this);

      int d=r.ndims();
      GELIB_ASSRT(x.ndims()==d);
      GELIB_ASSRT(y.ndims()==d);

      int b=r.getb();
      GELIB_ASSRT(x.getb()==b);
      GELIB_ASSRT(y.getb()==b);

      auto gdims=r.get_gdims();
      GELIB_ASSRT(x.get_gdims()==gdims);
      GELIB_ASSRT(y.get_gdims()==gdims);

      if(d==3){
	r.for_each_batch_multi(x,y,[&](const int b, const TENSOR& r, const TENSOR& x, const TENSOR& y){
	    lambda(r,x,y);
	  });
	return;
      }

      if(d==5){
	int g0=r.dims[1];
	int g1=r.dims[2];
	r.for_each_batch_multi(x,y,[&](const int b, const TENSOR& r, const TENSOR& x, const TENSOR& y){
	    for(int _g0=0; _g0<g0; _g0++)
	      for(int _g1=0; _g1<g1; _g1++)
		lambda(r.slice(0,_g0).slice(0,_g1),x.slice(0,_g0).slice(0,_g1),y.slice(0,_g0).slice(0,_g1));
	  });
	return;
      }

      GELIB_ASSRT(false);
    }


  public: // ---- Promotions ---------------------------------------------------------------------------------


    void promote_grid_to(const int g){
      GELIB_ASSRT(ndims()==4);
      if(dims[1]==g) return;
      GELIB_ASSRT(dims[1]==1);
      dims[1]=g;
      strides[1]=0;
    }


    void promote_grid_to(const int g0, const int g1){
      int d=dims.size();
      if(d==3){
	dims=cnine::Gdims({dims[0],g0,g1,dims[1],dims[2]});
	strides=cnine::GstridesB(vector<size_t>({strides[0],0,0,strides[1],strides[2]}));
	return;
      }
      if(d==5){
	GELIB_ASSRT(dims[1]==g0);
	GELIB_ASSRT(dims[2]==g1);
	return;
      }
      GELIB_ERROR("The number of grid dimensions is not 0 or 2.");
    }


    void promote_grid_to(const Gdims& _gdims){
      int d=dims.size();
      int n=_gdims.size();
      GELIB_ASSRT(d==3);
      if(n==0) return;

      Gdims new_dims(n+3);
      new_dims[0]=dims[0];
      std::copy(_gdims.begin(),_gdims.end(),new_dims.begin()+1);
      new_dims[n+1]=dims[1];
      new_dims[n+2]=dims[2];
      dims=new_dims;

      GstridesB new_strides(n+3,cnine::fill_raw());
      new_strides[0]=strides[0];
      for(int i=0; i<n; i++)
	new_strides[i+1]=0;
      new_strides[n+1]=strides[1];
      new_strides[n+2]=strides[2];
      strides=new_strides;
    }




    bool reconcile_grids(Gpart& x, Gpart& y){
      Gdims common_gdims=gdims();
      if(x.dims.size()>3) common_gdims=x.gdims();
      if(y.dims.size()>3) common_gdims=y.gdims();
      if(common_gdims.size()==0) return true;
      
      if(dims.size()==3) promote_grid_to(common_gdims);
      else if(gdims()!=common_gdims) return false;

      if(x.dims.size()==3) x.promote_grid_to(common_gdims);
      else if(x.gdims()!=common_gdims) return false;

      if(y.dims.size()==3) y.promote_grid_to(common_gdims);
      else if(y.gdims()!=common_gdims) return false;

      return true;
    }


  public: // ---- Canonicalization ---------------------------------------------------------------------------


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


    void canonicalize_to_5d(){
      int d=dims.size();
      if(d==3){
	dims=cnine::Gdims({dims[0],1,1,dims[1],dims[2]});
	strides=cnine::GstridesB({strides[0],(size_t)0,(size_t)0,strides[1],strides[2]});
      }
      if(d==4){
	dims=cnine::Gdims({dims[0],dims[1],1,dims[2],dims[3]});
	strides=cnine::GstridesB({strides[0],strides[1],(size_t)0,strides[2],strides[3]});
      }
      if(d==5) return;
      if(d>5){
	Gdims gdims=dims.chunk(1,d-3);
	GstridesB gstrides=strides.chunk(1,d-3);
	auto [s,g]=gstrides.fuser(gdims);
	GELIB_ASSRT(g!=-1);
	dims=cnine::Gdims({dims[0],g,1,dims[d-2],dims[d-1]});
	strides=cnine::GstridesB({strides[0],(size_t)s,(size_t)0,strides[d-2],strides[d-1]});
      }
    }


    bool co_canonicalize_to_5d(Gpart& x, Gpart& y){
      int d=dims.size();
      GELIB_ASSRT(x.dims.size()==d);
      GELIB_ASSRT(y.dims.size()==d);
      if(d==5) return true;

      if(d>5){
	int n=d-3;
	vector<int> ordering(n);
	for(int i=0; i<n; i++)
	  ordering[i]=i;

	GstridesB rstrides=strides.chunk(1,n);
	if(rstrides.max()!=rstrides.min()) 
	  ordering=rstrides.ordering();

	GstridesB xstrides=x.strides.chunk(1,n);
	if(xstrides.max()!=xstrides.min()) 
	  ordering=xstrides.ordering();

	GstridesB ystrides=y.strides.chunk(1,n);
	if(ystrides.max()!=ystrides.min()) 
	  ordering=ystrides.ordering();

	Gdims rdims=dims.chunk(1,d-3);
	GELIB_ASSRT(x.dims.chunk(1,d-3)==rdims);
	GELIB_ASSRT(y.dims.chunk(1,d-3)==rdims);
	GELIB_ASSRT(rstrides.fusible(rdims,ordering));
	GELIB_ASSRT(xstrides.fusible(rdims,ordering));
	GELIB_ASSRT(ystrides.fusible(rdims,ordering));
      }

      canonicalize_to_5d();
      x.canonicalize_to_5d();
      y.canonicalize_to_5d();
      return true;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    GPART gather(const cnine::GatherMapB& gmap, const int d=0) const{
      Gdims adims=gdims();
      GELIB_ASSRT(d<adims.size());
      GELIB_ASSRT(gmap.n_in==adims(d));
      adims[d]=gmap.n_out;
      auto R=GPART::zeros_like(*this,adims);
      R.add_gather(*this,gmap,d);
      return R;
    }

    GPART gather_back(const cnine::GatherMapB& gmap, const int d=0) const{
      Gdims adims=gdims();
      GELIB_ASSRT(d<adims.size());
      GELIB_ASSRT(gmap.n_out==adims(d));
      adims[d]=gmap.n_in;
      auto R=GPART::zeros_like(*this,adims);
      R.add_gather_back(*this,gmap,d);
      return R;
    }

    void add_gather(const GPART& x, const cnine::GatherMapB& gmap, const int d=0){
      cnine::GatherSlices()(*this,x,gmap,d+1);
    }

    void add_gather_back(const GPART& r, const cnine::GatherMapB& gmap, const int d=0){
      cnine::GatherSlices()(*this,r,gmap.inv(),d+1);
    }

 
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


    void add_CGproduct(GPART x, GPART y, const int offs=0){
      GPART r(*this);
      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(y.get_dev()==dev);

      if(!r.reconcile_batches(x,y))
	GELIB_NONFATAL("Skipping CGproduct: batch dimensions cannot be reconciled.");

      if(!r.reconcile_grids(x,y))
	GELIB_NONFATAL("Skipping CGproduct: grid dimensions cannot be reconciled.");

      r.co_canonicalize_to_5d(x,y);

      if(dev==0){
	auto C=r.get_CGmatrix(x,y);
	r.for_each_cell_multi(x,y,[&](const TENSOR& r, const TENSOR& x, const TENSOR& y){
	    GPART::add_CGproduct_kernel(r,x,y,C,offs);
	      });
      }

      if(dev==1){
	GPART::add_CGproduct_dev(r,x,y,offs);
      }

    }


    void add_CGproduct_back0(GPART r, GPART y, const int offs=0){
      GPART x(*this);
      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(y.get_dev()==dev);

      if(!r.reconcile_batches(x,y))
	GELIB_NONFATAL("Skipping CGproduct: batch dimensions cannot be reconciled.");

      if(!r.reconcile_grids(x,y))
	GELIB_NONFATAL("Skipping CGproduct: grid dimensions cannot be reconciled.");

      r.co_canonicalize_to_5d(x,y);

      if(dev==0){
	auto C=r.get_CGmatrix(x,y);
	x.for_each_cell_multi(r,y,[&](const TENSOR& x, const TENSOR& r, const TENSOR& y){
	    GPART::add_CGproduct_back0_kernel(r,x,y,C,offs);});
      }

      if(dev==1){
	GPART::add_CGproduct_back0_dev(r,x,y,offs);
      }

    }


    void add_CGproduct_back1(GPART r, GPART x, const int offs=0){
      GPART y(*this);
      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(y.get_dev()==dev);

      if(!r.reconcile_batches(x,y))
	GELIB_NONFATAL("Skipping CGproduct: batch dimensions cannot be reconciled.");

      if(!r.reconcile_grids(x,y))
	GELIB_NONFATAL("Skipping CGproduct: grid dimensions cannot be reconciled.");

      r.co_canonicalize_to_5d(x,y);

      if(dev==0){
	auto C=r.get_CGmatrix(x,y);
	y.for_each_cell_multi(r,x,[&](const TENSOR& y, const TENSOR& r, const TENSOR& x){
	    GPART::add_CGproduct_back1_kernel(r,x,y,C,offs);});
      }

      if(dev==1){
	GPART::add_CGproduct_back1_dev(r,x,y,offs);
      }

    }


  public: // ---- Diagonal CG-products -----------------------------------------------------------------------


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


    void add_DiagCGproduct(GPART x, GPART y, const int offs=0){
      GPART r(*this);
      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(y.get_dev()==dev);

      if(!r.reconcile_batches(x,y))
	GELIB_NONFATAL("Skipping DiagCGproduct: batch dimensions cannot be reconciled.");

      if(!r.reconcile_grids(x,y))
	GELIB_NONFATAL("Skipping DiagCGproduct: grid dimensions cannot be reconciled.");

      r.co_canonicalize_to_5d(x,y);

      if(dev==0){
	auto C=r.get_CGmatrix(x,y);
	r.for_each_cell_multi(x,y,[&](const TENSOR& r, const TENSOR& x, const TENSOR& y){
	    GPART::add_DiagCGproduct_kernel(r,x,y,C,offs);
	      });
      }

      if(dev==1){
	GPART::add_DiagCGproduct_dev(r,x,y,offs);
      }

    }


    void add_DiagCGproduct_back0(GPART r, GPART y, const int offs=0){
      GPART x(*this);
      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(y.get_dev()==dev);

      if(!r.reconcile_batches(x,y))
	GELIB_NONFATAL("Skipping DiagCGproduct: batch dimensions cannot be reconciled.");

      if(!r.reconcile_grids(x,y))
	GELIB_NONFATAL("Skipping DiagCGproduct: grid dimensions cannot be reconciled.");

      r.co_canonicalize_to_5d(x,y);

      if(dev==0){
	auto C=r.get_CGmatrix(x,y);
	x.for_each_cell_multi(r,y,[&](const TENSOR& x, const TENSOR& r, const TENSOR& y){
	    GPART::add_DiagCGproduct_back0_kernel(r,x,y,C,offs);});
      }

      if(dev==1){
	GPART::add_DiagCGproduct_back0_dev(r,x,y,offs);
      }

    }


    void add_DiagCGproduct_back1(GPART r, GPART x, const int offs=0){
      GPART y(*this);
      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(y.get_dev()==dev);

      if(!r.reconcile_batches(x,y))
	GELIB_NONFATAL("Skipping DiagCGproduct: batch dimensions cannot be reconciled.");

      if(!r.reconcile_grids(x,y))
	GELIB_NONFATAL("Skipping DiagCGproduct: grid dimensions cannot be reconciled.");

      r.co_canonicalize_to_5d(x,y);

      if(dev==0){
	auto C=r.get_CGmatrix(x,y);
	y.for_each_cell_multi(r,x,[&](const TENSOR& y, const TENSOR& r, const TENSOR& x){
	    GPART::add_DiagCGproduct_back1_kernel(r,x,y,C,offs);});
      }

      if(dev==1){
	GPART::add_DiagCGproduct_back1_dev(r,x,y,offs);
      }

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
      oss<<indent<<static_cast<const GPART&>(*this).repr()<<":"<<endl;
      oss<<str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gpart& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 



