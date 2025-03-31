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

#ifndef _GElibGelement
#define _GElibGelement

#include "GElib_base.hpp"
//#include "GelementObj.hpp"


namespace GElib{


  class Gelement{
  public:



  public: // ---- Access ---------------------------------------------------------------------------------


  public: // ---- Operations ---------------------------------------------------------------------------------

   

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Gelement";
    }

    string repr() const{
      return "";
      //return obj->repr();
    }
    
    string str(const string indent="") const{
      return "";
      //return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Gelement& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
