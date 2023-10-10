// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _EndMap
#define _EndMap

#include <cnine/tensors>
//#include "Tensor.hpp"
//#include "TensorFunctions.hpp"
#include "Gtype.hpp"


namespace GElib{

  template<typename GROUP, typename TYPE>
  class EndMap{
  public:

    typedef cnine::Tensor<TYPE> _Tensor;
    typedef typename GROUP::IrrepIx _IrrepIx;

    map<_IrrepIx,_Tensor> maps;

    ~EndMap(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    EndMap(){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    EndMap(const Gtype<GROUP>& x, const FILLTYPE& dummy, const int _dev=0){
      for(auto& p:x){
	maps[p.first]=_Tensor({p.second,p.second},dummy,_dev);
      }
    }

    template<typename SPACE, typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    EndMap(const SPACE& x, const FILLTYPE& dummy, const int _dev=0):
      EndMap(x.get_tau(),dummy,_dev){}

    //for(auto& p:x.isotypics){
    //maps[p.first]=_Tensor(cnine::Gdims({p.second->m,p.second->m}),dummy,_dev);
    //}
    //}



  public: // ---- Named constructors ------------------------------------------------------------------------


    template<typename SPACE>
    static EndMap zero(const SPACE& x, const int _dev=0){
      return EndMap(x,cnine::fill_zero(),_dev);
    }

    template<typename SPACE>
    static EndMap identity(const SPACE& x, const int _dev=0){
      return EndMap(x,cnine::fill_identity(),_dev);
    }

    template<typename SPACE>
    static EndMap sequential(const SPACE& x, const int _dev=0){
      return EndMap(x,cnine::fill_sequential(),_dev);
    }

    template<typename SPACE>
    static EndMap gaussian(const SPACE& x, const SPACE& y, const int _dev=0){
      return EndMap(x,y,cnine::fill_gaussian(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    EndMap(const cnine::Transpose<EndMap>& x){
      for(auto& p:x.obj.maps)
	maps[p.first]=p.second.transp();
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    Gtype<GROUP> get_tau() const{
      Gtype<GROUP> R;
      for(auto& p:maps)
	R[p.first]=p.second.dim(0);
      return R;
    }

    void for_each(std::function<void(const _IrrepIx&, const _Tensor&)>& lambda) const{
      for(auto& p:maps)
	lambda(p.first,p.second);
    }


  public: // ---- Operations --------------------------------------------------------------------------------

    
    const _Tensor& operator[](const _IrrepIx& ix) const{
      return const_cast<EndMap<GROUP,TYPE>&>(*this).maps[ix];
    }

    EndMap conjugate(const EndMap& T) const{
      return cnine::transp(T)*(*this)*T;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------

    
   string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:maps){
	oss<<indent<<"Component "<<p.first<<":"<<endl;
	oss<<p.second.str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const EndMap& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename GROUP, typename TYPE>
  inline EndMap<GROUP,TYPE> operator+(const EndMap<GROUP,TYPE>& x, const EndMap<GROUP,TYPE>& y){
    EndMap<GROUP,TYPE> R;
    for(auto& p:x.maps)
      R.maps.emplace(p.first,p.second+y[p.first]);
    //R.maps[p.first]=p.second+y[p.first];
    return R;
  }

  template<typename GROUP, typename TYPE>
  inline EndMap<GROUP,TYPE> operator*(const EndMap<GROUP,TYPE>& x, const EndMap<GROUP,TYPE>& y){
    EndMap<GROUP,TYPE> R;
    for(auto& p:x.maps)
      R.maps[p.first]=p.second*y[p.first];
    return R;
  }

  template<typename GROUP, typename TYPE>
  inline EndMap<GROUP,TYPE> operator*(const EndMap<GROUP,TYPE>& x, const cnine::Transpose<EndMap<GROUP,TYPE> >& y){
    EndMap<GROUP,TYPE> R;
    for(auto& p:x.maps)
      R.maps[p.first]=p.second*cnine::transp(y.obj[p.first]);
    return R;
  }

  template<typename GROUP, typename TYPE>
  inline EndMap<GROUP,TYPE> operator*(const cnine::Transpose<EndMap<GROUP,TYPE> >& x, const EndMap<GROUP,TYPE>& y){
    EndMap<GROUP,TYPE> R;
    for(auto& p:x.obj.maps)
      R.maps[p.first]=cnine::transp(p.second)*y[p.first];
    return R;
  }

  template<typename GROUP, typename TYPE>
  inline EndMap<GROUP,TYPE> tprod(const EndMap<GROUP,TYPE>& x, const EndMap<GROUP,TYPE>& y){
    EndMap<GROUP,TYPE> R(tprod(x.get_tau(),y.get_tau()),cnine::fill_zero());
    Gtype<GROUP> offs;
    for(auto& p:x.maps)
      for(auto& q:y.maps){
	cnine::Tensor<TYPE> T=tprod(p.second,q.second);
	GROUP::for_each_CGcomponent(p.first,q.first,[&](const typename GROUP::IrrepIx& ix, const int m){
	    GELIB_ASSRT(m==1);
	    int width=m*p.second.dim(0)*q.second.dim(0);
	    R[ix].block({width,width},{offs[ix],offs[ix]})=T;
	    offs[ix]+=width;
	  });
      }
    return R;
  }

}

#endif 
