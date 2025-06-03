/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineLtensor
#define _CnineLtensor

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "DimLabels.hpp"
#include "LtensorSpec.hpp"
#include "NamedTypes.hpp"
#include "MemoryManager.hpp"


namespace cnine{


  /*
  using BatchArgument=NamedType<int, struct BatchArgumentTag>;
  using GridArgument=NamedType<Gdims, struct GridArgumentTag>;
  using DimsArgument=NamedType<Gdims, struct DimsArgumentTag>;
  using ChannelsArgument=NamedType<int, struct ChannelsArgumentTag>;
  using FillArgument=NamedType<int, struct FillArgumentTag>;
  using DeviceArgument=NamedType<int, struct DeviceArgumentTag>;

  static const BatchArgument::argument batch;
  static const GridArgument::argument grid;
  static const DimsArgument::argument cdims;
  static const ChannelsArgument::argument channels;
  static const FillArgument::argument filltype;
  static const DeviceArgument::argument device;
  */


  template<typename TYPE>
  class Ltensor;

  inline Itensor2_view batch_grid_fused_view2_of(const Ltensor<int>& x);
  inline Rtensor2_view batch_grid_fused_view2_of(const Ltensor<float>& x);
  inline Ctensor2_view batch_grid_fused_view2_of(const Ltensor<complex<float> >& x);
  inline Itensor3_view batch_grid_fused_view3_of(const Ltensor<int>& x);
  inline Rtensor3_view batch_grid_fused_view3_of(const Ltensor<float>& x);
  inline Ctensor3_view batch_grid_fused_view3_of(const Ltensor<complex<float> >& x);



  template<typename TYPE>
  class Ltensor: public TensorView<TYPE>{
  public:

    enum Ttype{batch_grid_cell};

    typedef TYPE<TensorView> BASE;

    //using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::reset;
    using BASE::set_zero;

    DimLabels labels;


  public: // ---- Constructors without labels ----------------------------------------------------------------


    Ltensor(): 
      Ltensor({1},0,0){}
      //Ltensor({1},DimLabels(),0,0){}

    Ltensor(const Gdims& _dims):
      BASE(_dims,0,0){}

    Ltensor(const Gdims& _dims, const int fcode, const int _dev=0):
      BASE(_dims,fcode,_dev){}

    Ltensor(const MemArr<TYPE>& _arr, const Gdims& _dims):
      BASE(_arr,_dims,GstridesB(_dims)){}

    Ltensor(TYPE* _arr, const Gdims& _dims, const int _dev=0):
      Ltensor(MemArr<TYPE>(_arr,_dev),_dims){}


  public: // ---- Constructors with labels -------------------------------------------------------------------


    Ltensor(const Gdims& _dims, const DimLabels& _labels, const int fcode, const int _dev=0):
      BASE(_dims,fcode,_dev), 
      labels(_labels){}

    Ltensor(const int _b, const Gdims& _gdims, const Gdims& _ddims, const int fcode, const int _dev=0):
      BASE(Gdims(_b,_gdims,_ddims),fcode,_dev), 
      labels(_b,_gdims.size()){}

    Ltensor(const MemArr<TYPE>& _arr, const Gdims& _dims, const GstridesB& _strides, const DimLabels& _labels):
      BASE(_arr,_dims,_strides),
      labels(_labels){}


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int b=0;
      Gdims gdims;
      Gdims cdims;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ltensor(const Gdims& _cdims, const Args&... args){
      vparams v;
      v.cdims=_cdims;
      unroller(v,args...);
      reset(Ltensor(v.b,v.gdims,v.cdims,v.fcode,v.dev));
    }

    template<typename... Args>
    void unroller(vparams& v, const cnine::BatchArgument& x, const Args&... args){
      v.b=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::GridArgument& x, const Args&... args){
      v.gdims=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::DimsArgument& x, const Args&... args){
      v.cdims=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::FillArgument& x, const Args&... args){
      v.fcode=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::DeviceArgument& x, const Args&... args){
      v.dev=x.get(); unroller(v, args...);}

    void unroller(vparams& v){}


  public: // ---- LtensorSpec --------------------------------------------------------------------------------


    //[[deprecated]]
    Ltensor(const LtensorSpec<TYPE>& g):
      Ltensor(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}

    static LtensorSpec<TYPE> make() {return LtensorSpec<TYPE>();}
    static LtensorSpec<TYPE> raw() {return LtensorSpec<TYPE>().raw();}
    static LtensorSpec<TYPE> zero() {return LtensorSpec<TYPE>().zero();}
    static LtensorSpec<TYPE> sequential() {return LtensorSpec<TYPE>().sequential();}
    static LtensorSpec<TYPE> gaussian() {return LtensorSpec<TYPE>().gaussian();}
    
    LtensorSpec<TYPE> spec() const{
      return LtensorSpec<TYPE>(dims,labels,dev);
    }


  public: // ---- Other constructors ------------------------------------------------------------------------


    Ltensor(const initializer_list<initializer_list<TYPE> >& list, const int _dev=0){
      int n0=list.size();
      CNINE_ASSRT(n0>0);
      int n1=list.begin()->size();
      Ltensor<TYPE> T(Gdims(n0,n1)); 
      int i=0;
      for(auto& p: list){
	int j=0;
	for(auto& q: p)
	  T.set(i,j++,q);
	i++;
      }
      if(_dev>0) T.move_to_device(_dev);
      reset(T);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    Ltensor(const Ltensor& x):
      BASE(x),
      labels(x.labels){}

    Ltensor& operator=(const Ltensor& x){
      BASE::operator=(x);
      labels=x.labels;
      return *this;
    }
    
    Ltensor copy() const{
      FNTRACE();
      Ltensor R(dims,labels,0,dev);
      R=*this;
      return R;
    }

    Ltensor copy(MemoryManager* mm) const{
      FNTRACE();
      using_vram_manager vv(mm);
      Ltensor R(dims,labels,0,dev);
      R=*this;
      return R;
    }

   Ltensor copy(const int _dev) const{
      FNTRACE();
      Ltensor R(dims,labels,0,_dev);
      R=*this;
      return R;
    }

    Ltensor zeros_like() const{
      FNTRACE();
      return Ltensor(dims,labels,0,dev);
    }

    Ltensor gaussian_like() const{
      FNTRACE();
      return Ltensor(dims,labels,4,dev);
    }

    Ltensor like(TYPE* _arr) const{
      FNTRACE();
      return Ltensor(_arr,dims,dev);
    }


  public: // ---- Views -------------------------------------------------------------------------------------


    //auto batch_grid_fused_view1() const -> decltype(batch_grid_fused_view1_of(*this)){
    //CNINE_ASSRT(ndims()==1);
    //return batch_grid_fused_view1_of(*this);
    //}

    auto batch_grid_fused_view2() const -> decltype(batch_grid_fused_view2_of(*this)){
      return batch_grid_fused_view2_of(*this);
    }

    auto bgfused_view3() const -> decltype(batch_grid_fused_view3_of(*this)){
      return batch_grid_fused_view3_of(*this);
    }


  public: // ---- Memory managed ----------------------------------------------------------------------------


    Ltensor(const MemoryManager& manager, const Gdims& _dims, const int fcode, const int _dev){
      CNINE_ASSRT(fcode<2);
      FNTRACE();
      arr=MemArr<TYPE>(manager,_dims.total(),_dev);
      dims=_dims;
      strides=GstridesB(_dims);
      dev=_dev;
      if(fcode==0) set_zero();
    }

    static Ltensor<TYPE> vram_managed(MemoryManager* mm, const Gdims& _dims, const int fcode, const int _dev){
      using_vram_manager vv(mm);
      return Ltensor<TYPE>(_dims,fcode,_dev);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    Ltensor(const TensorView<TYPE>& x):
      BASE(x){}

    Ltensor(TYPE* _arr, const int _dev, const Gdims& _dims, const GstridesB& _strides):
      TensorView<TYPE>(new MemBlob<TYPE>(_arr,_dev),_dims,_strides){
    }

#ifdef _WITH_ATEN
    Ltensor<TYPE>(const at::Tensor& T):
      BASE(T){}

    // this is pretty dangerous 
    static Ltensor view(const at::Tensor& T){
      return Ltensor(T.data<TYPE>(),T.type().is_cuda(),Gdims(T),GstridesB(T));
    }

#endif 


  public: // ---- Access ------------------------------------------------------------------------------------


    Ttype ttype() const{
      return batch_grid_cell;
    }

    bool batch_grid_regular() const{
      return true;
    }


  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return labels._batched;
    }

    int nbatch() const{
      if(is_batched()) return dims[0];
      else return 0;
    }

    Ltensor batch(const int i) const{
      CNINE_ASSRT(is_batched());
      CNINE_CHECK_RANGE(dims.check_in_range_d(0,i,string(__PRETTY_FUNCTION__)));
      return Ltensor(arr+strides[0]*i,dims.chunk(1),strides.chunk(1),labels.copy().set_batched(false));
    }

    void for_each_batch(const std::function<void(const int, const Ltensor& x)>& lambda) const{
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_grid() const{
      return labels._narray>0;
    }

    int ngdims() const{
      return labels._narray;
    }

    int nbgdims() const{
      return labels._batched+labels._narray;
    }

    Gdims gdims() const{
      return labels.gdims(dims);
    }

    Gdims gdim(const int i) const{
      return labels.gdims(dims)[i];
    }

    int total_bgdims() const{
      if(nbgdims()==0) return 1;
      return dims[0]*strides[0]/strides[nbgdims()-1];
    }

    GstridesB gstrides() const{
      return labels.gstrides(strides);
    }

    int min_gstride() const{
      if(nbgdims()==0) return 0;
      return strides[nbgdims()-1];
    }

    Ltensor cell(const Gindex& ix) const{
      CNINE_ASSRT(!is_batched());
      CNINE_ASSRT(ix.size()==labels._narray);
      if(is_batched()) 
	return Ltensor(arr+gstrides().offs(ix),bcdims(),bcstrides(),labels.copy().set_ngrid(0));
      else 
	return Ltensor(arr+gstrides().offs(ix),cdims(),cstrides(),labels.copy().set_ngrid(0));
    }

    Ltensor cell(const int b, const Gindex& ix) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(b<nbatch());
      CNINE_ASSRT(ix.size()==labels._narray);
      return Ltensor(arr+strides[0]*b+gstrides().offs(ix),cdims(),cstrides(),labels.copy().set_batched(false).set_ngrid(0));
    }

    void for_each_cell(const std::function<void(const Gindex&, const Ltensor& x)>& lambda) const{
      CNINE_ASSRT(!is_batched());
      gdims().for_each_index([&](const vector<int>& ix){
	  lambda(ix,cell(ix));
	});
    }
    
    void for_each_cell(const std::function<void(const int b, const Gindex&, const Ltensor& x)>& lambda) const{
      CNINE_ASSRT(is_batched());
      for(int b=0; b<nbatch(); b++)
	gdims().for_each_index([&](const vector<int>& ix){
	    lambda(b,ix,cell(b,ix));
	});
    }
    

  public: // ---- Cells --------------------------------------------------------------------------------------


    int ncdims() const{
      return dims.size()-labels._narray-labels._batched;
    }

    Gdims cdims() const{
      return labels.cdims(dims);
    }

    int cdim(const int i) const{
      CNINE_ASSRT(i+nbgdims()<dims.size());
      return dims[nbgdims()+i];
    }

    GstridesB cstrides() const{
      return labels.cstrides(strides);
    }

    int cstride(const int i) const{
      CNINE_ASSRT(i+nbgdims()<dims.size());
      return strides[nbgdims()+i];
    }


  public: // ---- Batched cells ------------------------------------------------------------------------------


    Gdims bcdims() const{
      return labels.bcdims(dims);
    }

    GstridesB bcstrides() const{
      return labels.bcstrides(strides);
    }

    Ltensor batched_cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==labels._narray);
      return Ltensor(arr+gstrides().offs(ix),bcdims(),bcstrides(),labels.copy().set_ngrid(0));
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    Ltensor operator*(const TYPE c) const{
      Ltensor R=zeros_like();
      R.add(*this,c);
      return R;
    }

    Ltensor mprod(const Ltensor& y) const{
      return mult(*this,y);
    }

    Ltensor operator*(const Ltensor& y) const{
      return mult(*this,y);
    }

    Ltensor scale_columns(const Ltensor& y) const{
      Ltensor R=zeros_like();
      R.add_scale_columns(*this,y);
      return R;
    }

    Ltensor sum(const int d) const{
      Ltensor R(dims.remove(d),0,dev);
      R.add_sum(d,*this);
      return R;
    }

    Ltensor ReLU(const TYPE alpha=0.1) const{
      Ltensor R=zeros_like();
      R.add_ReLU(*this,alpha);
      return R;
    }


  public: // ---- Stacking ----------------------------------------------------------------------------------


    template<typename OBJ>
    static Ltensor stack(int d, const vector<OBJ>& list){
      CNINE_ASSRT(list.size()>0);
      CNINE_ASSRT(d<list[0].ndims());
      Gdims dims0=list[0].dims;
      Gdims rem=list[0].dims.remove(d);
      int t=0;
      for(int i=0; i<list.size(); i++){
	t+=list[i].dim(d);
	CNINE_ASSRT(list[i].dims.remove(d)==rem);
      }
      Ltensor R(dims0.set(d,t),0,list[0].get_dev());
      t=0;
      for(int i=0; i<list.size(); i++){
	R.slices(d,t,list[i].dim(d))+=list[i];
	t+=list[i].dim(d);
      }
      return R;
    }

    template<typename OBJ>
    static Ltensor stack(int d, const vector<reference_wrapper<OBJ> >& list){
      CNINE_ASSRT(list.size()>0);
      CNINE_ASSRT(d<list[0].get().ndims());
      Gdims dims0=list[0].get().dims.remove(d);
      int t=0;
      for(int i=0; i<list.size(); i++){
	t+=list[i].get().dim(d);
	CNINE_ASSRT(list[i].get().dims.remove(d)==dims0);
      }
      Ltensor R(dims0.set(d,t),0,list[0].get().get_dev());
      t=0;
      for(int i=0; i<list.size(); i++){
	R.slices(d,t,list[i].get().dim(d))+=list[i].get();
	t+=list[i].get().dim(d);
      }
      return R;
    }

    template<typename OBJ>
    static Ltensor stack(int d, const initializer_list<OBJ>& list){
      vector<Ltensor<TYPE> > x;
      for(auto& p:list) x.push_back(p);
      return stack(0,x);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Ltensor";
    }

    string repr() const{
      ostringstream oss;
      //oss<<"Ltensor"<<labels.str(dims); //<<"["<<dev<<"]";
      oss<<"Ltensor(";
      if(is_batched()) oss<<"b="<<nbatch()<<",";
      if(is_grid()) oss<<"grid="<<gdims()<<",";
      oss<<"dim="<<cdims()<<",";
      if(dev>0) oss<<"dev="<<dev<<",";
      oss<<"\b)";
      return oss.str();
    }

    string to_string(const string indent="") const{

      if(is_batched()){
	ostringstream oss;
	for_each_batch([&](const int b, const Ltensor& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.to_string(indent+"  ");
	  });
	return oss.str();
      }
      
      if(is_grid()){
	ostringstream oss;
	for_each_cell([&](const Gindex& ix, const Ltensor& x){
	    oss<<indent<<"Cell"<<ix<<":"<<endl;
	    oss<<x.to_string(indent+"  ");
	  });
	return oss.str();
      }
      
      return BASE::str(indent);
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<to_string(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ltensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename TYPE>
  class LtensorView: public Ltensor<TYPE>{
  public:
    
    typedef Ltensor<TYPE> BASE;

    LtensorView(TYPE* _arr, const int _dev, const Gdims& _dims):
      BASE(_arr,_dev,_dims,GstridesB(_dims)){}

    ~LtensorView(){
      BASE::arr.blob->arr=nullptr;
    }

  };


  // ---- View converters ----------------------------------------------------


  inline Itensor2_view batch_grid_fused_view2_of(const Ltensor<int>& x){
    CNINE_ASSRT(x.ttype()==Ltensor<int>::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==1);
    return Itensor2_view(x.mem(),x.total_bgdims(),x.cdim(0),x.min_gstride(),x.cstride(0),x.dev);
  }

  inline Rtensor2_view batch_grid_fused_view2_of(const Ltensor<float>& x){
    CNINE_ASSRT(x.ttype()==Ltensor<float>::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==1);
    return Rtensor2_view(x.mem(),x.total_bgdims(),x.cdim(0),x.min_gstride(),x.cstride(0),x.dev);
  }

  inline Ctensor2_view batch_grid_fused_view2_of(const Ltensor<complex<float> >& x){
    CNINE_ASSRT(x.ttype()==Ltensor<complex<float> >::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==1);
    return Ctensor2_view(x.arr.ptr_as<float>(),x.arr.ptr_as<float>()+1,
      x.total_bgdims(),x.cdim(0),2*x.min_gstride(),2*x.cstride(0),x.dev);
  }


  inline Itensor3_view batch_grid_fused_view3_of(const Ltensor<int>& x){
    CNINE_ASSRT(x.ttype()==Ltensor<int>::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==2);
    return Itensor3_view(x.mem(),x.total_bgdims(),x.cdim(0),x.cdim(1),x.min_gstride(),x.cstride(0),x.cstride(1),x.dev);
  }

  inline Rtensor3_view batch_grid_fused_view3_of(const Ltensor<float>& x){
    CNINE_ASSRT(x.ttype()==Ltensor<float>::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==2);
    return Rtensor3_view(x.mem(),x.total_bgdims(),x.cdim(0),x.cdim(1),x.min_gstride(),x.cstride(0),x.cstride(1),x.dev);
  }

  inline Ctensor3_view batch_grid_fused_view3_of(const Ltensor<complex<float> >& x){
    CNINE_ASSRT(x.ttype()==Ltensor<complex<float> >::batch_grid_cell);
    CNINE_ASSRT(x.ncdims()==2);
    return Ctensor3_view(x.arr.ptr_as<float>(),x.arr.ptr_as<float>()+1,
      x.total_bgdims(),x.cdim(0),x.cdim(1),2*x.min_gstride(),2*x.cstride(0),2*x.cstride(1),x.dev);
  }

  template<typename TYPE>
  inline Ltensor<TYPE> operator*(const Ltensor<TYPE>& x, const Ltensor<TYPE>& y){
    Ltensor<TYPE> R(x.get_dims().Mprod(y.get_dims()),0,x.get_dev());
    R.add_mprod(x,y);
    return R;
  }

  template<typename TYPE>
  inline Ltensor<TYPE> operator*(const TensorView<TYPE>& x, const Ltensor<TYPE>& y){
    Ltensor<TYPE> R(x.get_dims().Mprod(y.get_dims()),0,x.get_dev());
    R.add_mprod(x,y);
    return R;
  }

  inline Ltensor<complex<float> > mult(const Ltensor<complex<float> >& x, const Ltensor<complex<float> >& y){
    Gdims d(x.dims);
    d.set_back(y.dims.back());
    Ltensor<complex<float> > r(d,x.labels,0,x.dev);
    Ctensor2_view rv(r.arr.ptr_as<float>(),r.arr.ptr_as<float>()+1,r.total_bgdims()*r.cdim(0),r.cdim(1),2*r.cstride(0),2*r.cstride(1),r.dev);
    Ctensor2_view xv(x.arr.ptr_as<float>(),x.arr.ptr_as<float>()+1,x.total_bgdims()*x.cdim(0),x.cdim(1),2*x.cstride(0),2*x.cstride(1),x.dev);
    Ctensor2_view yv(y.arr.ptr_as<float>(),y.arr.ptr_as<float>()+1,y.cdim(0),y.cdim(1),2*y.cstride(0),2*y.cstride(1),y.dev);
    rv.add_matmul_AA(xv,yv);
    return r;
  }
  
  inline Ltensor<float> mult(const Ltensor<float>& x, const Ltensor<float>& y){
    Gdims d(x.dims);
    d.set_back(y.dims.back());
    Ltensor<float> r(d,x.labels,0,x.dev);
    Rtensor2_view rv(r.arr.ptr(),r.total_bgdims()*r.cdim(0),r.cdim(1),r.cstride(0),r.cstride(1),r.dev);
    Rtensor2_view xv(x.arr.ptr(),x.total_bgdims()*x.cdim(0),x.cdim(1),x.cstride(0),x.cstride(1),x.dev);
    Rtensor2_view yv(y.arr.ptr(),y.cdim(0),y.cdim(1),y.cstride(0),y.cstride(1),y.dev);
    rv.add_matmul_AA(xv,yv);
    return r;
  }
  

}


namespace std{

  template<typename TYPE>
  struct hash<cnine::Ltensor<TYPE> >{
  public:
    size_t operator()(const cnine::Ltensor<TYPE>& x) const{
      size_t t=hash<cnine::Gdims>()(x.dims);
      if(x.is_regular()){
	int N=x.asize();
	for(int i=0; i<N; i++)
	  t=(t^hash<TYPE>()(x.arr[i]))<<1;
      }else{
	x.for_each([&t](const cnine::Gindex& ix, const TYPE v){
	    t=(t^hash<TYPE>()(v))<<1;});
      }
      return t;
    }
  };
}


#endif 
  /*
  inline Ltensor<float> mult(const bool condition, const MemoryManager* mm, const Ltensor<float>& x, const Ltensor<float>& y){
    Gdims d(x.dims);
    d.set_back(y.dims.back());
    Ltensor<float> r(condition,mm,d,x.labels,0,x.dev);
    Rtensor2_view rv(r.arr.ptr(),r.total_bgdims()*r.cdim(0),r.cdim(1),r.cstride(0),r.cstride(1),r.dev);
    Rtensor2_view xv(x.arr.ptr(),x.total_bgdims()*x.cdim(0),x.cdim(1),x.cstride(0),x.cstride(1),x.dev);
    Rtensor2_view yv(y.arr.ptr(),y.cdim(0),y.cdim(1),y.cstride(0),y.cstride(1),y.dev);
    rv.add_matmul_AA(xv,yv);
    return r;
  }
  */
    /*
    Ltensor(const MemoryManager* manager, const Gdims& _dims, const int fcode, const int _dev){
      CNINE_ASSRT(fcode<2);
      FNTRACE();
      dims=_dims;
      strides=GstridesB(_dims);
      dev=_dev;
      if(manager && _dev>0) 
	arr=MemArr<TYPE>(*manager,_dims.total(),_dev);
      else arr=MemArr<TYPE>(_dims.total(),_dev);
      if(fcode==0) set_zero();
    }
    */

    /*
    Ltensor(const bool condition, const MemoryManager* manager, const DimLabels& _labels, const Gdims& _dims, const int fcode, const int _dev){
      CNINE_ASSRT(fcode<2);
      FNTRACE();
      dims=_dims;
      labels=_labels;
      strides=GstridesB(_dims);
      dev=_dev;
      if(manager && condition) 
	arr=MemArr<TYPE>(*manager,_dims.total(),_dev);
      else MemArr<TYPE>(_dims.total(),_dev);
      if(fcode==0) set_zero();
    }
    */
