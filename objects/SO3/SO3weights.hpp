
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3weights
#define _SO3weights


namespace GElib{

  class SO3weights: public cnine::CtensorPackObj{
  public:

    typedef cnine::CtensorPackObj ctenspack;

    using ctenspack::ctenspack;

    //using CtensorPackObj::CtensorPackObj;


  public: // ---- Static constructors ------------------------------------------------------------------------


    static SO3weights raw(const SO3type& x, const SO3type& y, const int _dev=0){
      return CtensorPackObj(dimspack(x,y),cnine::fill_raw(),0,_dev);}

    static SO3weights zero(const SO3type& x, const SO3type& y, const int _dev=0){
      return CtensorPackObj(dimspack(x,y),cnine::fill_zero(),0,_dev);}

    static SO3weights gaussian(const SO3type& x, const SO3type& y, const int _dev=0){
      return CtensorPackObj(dimspack(x,y),cnine::fill_gaussian(),0,_dev);}

    
  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3weights(const CtensorPackObj& x):
      CtensorPackObj(x){}

    SO3weights(CtensorPackObj&& x):
      CtensorPackObj(std::move(x)){}


  private: // ------------------------------------------------------------------------------------------------


    static cnine::GdimsPack dimspack(const SO3type& x, const SO3type& y){
      assert(x.size()==y.size());
      cnine::GdimsPack R;
      for(int i=0; i<x.size(); i++)
	R.push_back(cnine::Gdims({x[i],y[i]}));
      return R;
    }

  };
}

#endif 
