
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElib_base
#define _GElib_base

#include <any>

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "NamedType.hpp"


using namespace std; 

#define _GELIB_VERSION "0.0.0 8/29/2024"

#define GELIB_ASSRT(condition) \
  if(!(condition)) throw std::runtime_error("GElib error in "+string(__PRETTY_FUNCTION__)+" : failed assertion "+#condition+".");

#define GELIB_ASSERT(condition, message) if (!(condition)) {cout<<message<<endl; assert ((condition)); exit(-1); }

#define GELIB_CHECK(condition,err) if(!condition) {{cnine::CoutLock lk; cerr<<"GElib error in function '"<<__PRETTY_FUNCTION__<<"' : "<<err<<endl;} exit(1);};
#define GELIB_UNIMPL() printf("GElib error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);
#define GELIB_ERROR(msg) throw std::runtime_error("GElib error in "+string(__PRETTY_FUNCTION__)+" : "+msg+".");

#define GELIB_NONFATAL(msg) {cnine::CoutLock lk; cerr<<"GElib exception in function '"<<__PRETTY_FUNCTION__<<"' : "<<msg<<endl;};


// ---- Helpers ------------------------------------------------------------------------------------------------------

namespace GElib{


  template<typename TYPE>
  auto print_if_possible(const TYPE& x, int) -> decltype(x.to_print(),std::to_string(1)){
    return x.to_print();
  }

  template<typename TYPE>
  string print_if_possible(const TYPE& x, long){
    return x.str();
  }

  template<typename TYPE>
  inline void print(const TYPE& x){
    cout<<print_if_possible(x,0)<<endl;
  }
  
  using IrrepArgument=cnine::NamedType<std::any, struct IrrepArgumentTag>;
  static const IrrepArgument::argument irrep;
  
  using TtypeArgument=cnine::NamedType<std::any, struct TtypeArgumentTag>;
  static const TtypeArgument::argument ttype;
  
}


// ---- CUDA STUFF ------------------------------------------------------------------------------------------


#define GELIB_CPUONLY() if(dev>0) {printf("Cengine error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}

#endif 
