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

#ifndef _GElibO3irrep
#define _GElibO3irrep

#include "GElib_base.hpp"
#include "O3element.hpp"
#include "O3index.hpp"
#include "SO3irrep.hpp"


namespace GElib{

  class O3irrep{
  public:

    O3index ix;
    
    O3irrep(const O3index& _ix): ix(_ix){}


  public: // ---- Operations ---------------------------------------------------------------------------------


    template<typename TYPE>
    cnine::TensorView<complex<TYPE> > matrix(const O3element<TYPE>& R) const{
      SO3irrep sub(ix.getl());
      int p=R.parity();
      if(ix.getp()==1) return sub.matrix<TYPE>(SO3element(R*(TYPE)p));
      else return sub.matrix<TYPE>(SO3element(R*(TYPE)p),p);
    }
    
    template<typename TYPE>
    cnine::TensorView<complex<TYPE> > matrix(const double alpha, const double beta, const double gamma, const int p) const{
      SO3irrep sub(ix.getl());
      if(ix.getp()==-1 && p==-1) return sub.matrix<TYPE>(alpha,beta,gamma,-1);
      else return sub.matrix<TYPE>(alpha,beta,gamma,1);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::O3irrep";
    }

    string repr() const{
      return "<O3irrep ix="+ix.str()+">";
    }

    string str(const string indent="") const{
      return "";
    }

    friend ostream& operator<<(ostream& stream, const O3irrep& x){
      stream<<x.str(); return stream;
    }


  };


}


#endif 
