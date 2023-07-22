// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _HomMap
#define _HomMap

#include "Tensor.hpp"
//#include "GSnSpaceObj.hpp"


namespace GElib{

  template<typename GROUP, typename TYPE>
  class HomMap{
  public:

    typedef cnine::Tensor<TYPE> _Tensor;
    typedef typename GROUP::IrrepIx _IrrepIx;

    map<_IrrepIx,_Tensor> maps;

    ~HomMap(){
      //for(auto& x: maps)
      //delete x.second;
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    HomMap(){}

    /*
    template<typename SPACE, typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    HomMap(const SPACE& x, const SPACE& y, const FILLTYPE& dummy, const int _dev=0){
      for(auto& p:x.isotypics){
	auto& q=const_cast<SPACE&>(y).isotypics[p.first]; // this is why we need a stash class 
	auto& X=*p.second;
	auto& Y=*q;
	maps[p.first]=_Tensor(cnine::Gdims({Y.m,X.m}),dummy,_dev);
      }
    }
    */


  public: // ---- Named constructors ------------------------------------------------------------------------


    template<typename SPACE>
    static HomMap zero(const SPACE& x, const SPACE& y, const int _dev=0){
      return HomMap(x,y,cnine::fill_zero(),_dev);
    }

    template<typename SPACE>
    static HomMap identity(const SPACE& x, const SPACE& y, const int _dev=0){
      return HomMap(x,y,cnine::fill_identity(),_dev);
    }

    template<typename SPACE>
    static HomMap sequential(const SPACE& x, const SPACE& y, const int _dev=0){
      return HomMap(x,y,cnine::fill_sequential(),_dev);
    }

    template<typename SPACE>
    static HomMap gaussian(const SPACE& x, const SPACE& y, const int _dev=0){
      return HomMap(x,y,cnine::fill_gaussian(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


  public: // ---- Access ------------------------------------------------------------------------------------



  public: // ---- Operations --------------------------------------------------------------------------------

    
    const _Tensor& operator[](const _IrrepIx& ix) const{
      return const_cast<HomMap<GROUP,TYPE>&>(*this).maps[ix];
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

    friend ostream& operator<<(ostream& stream, const HomMap& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename GROUP, typename TYPE>
  inline HomMap<GROUP,TYPE> operator*(const HomMap<GROUP,TYPE>& x, const HomMap<GROUP,TYPE>& y){
    HomMap<GROUP,TYPE> R;
    for(auto& p:x.maps)
      R.maps[p.first]=p.second*y.maps[p.first];
    return R;
  }

  template<typename GROUP, typename TYPE>
  inline HomMap<GROUP,TYPE> operator*(const HomMap<GROUP,TYPE>& x, const cnine::Transpose<HomMap<GROUP,TYPE> >& y){
    HomMap<GROUP,TYPE> R;
    for(auto& p:x.maps)
      R.maps[p.first]=p.second*cnine::transp(y.obj[p.first]);
    return R;
  }


}

#endif 
