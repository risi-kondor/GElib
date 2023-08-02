// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _CoupleIsotypics
#define _CoupleIsotypics

#include "cachedf.hpp"
#include "SnIsotypicSpace.hpp"
#include "IntegerPartition.hpp"
#include "BlockDiagonalize.hpp"


namespace GElib{


  template<typename TYPE>
  class CoupleIsotypics{
  public:

    typedef cnine::Tensor<TYPE> _Tensor;
    typedef Snob2::IntegerPartition IP;

    CoupleIsotypics(const map<IP,SnIsotypicSpace<TYPE> >& subs, vector<_Tensor>& spaces){
      int mult=subs.begin()->second.multiplicity();
      int pdim=subs.begin()->second.pdim();

      for(int i=0; i<spaces.size(); i++){
	cout<<4421<<endl;
	_Tensor& space=spaces[i];
	GELIB_ASSRT(space.dims[1]==pdim);

	for(auto& p: subs){
	  cout<<"Subrep "<<p.first<<endl;
	  cout<<p.second<<endl;
	  cout<<p.second.fuse01()*cnine::transp(spaces[i])<<endl;
	}
      }
    }

  };

}

#endif 
