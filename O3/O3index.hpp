/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2025, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _O3index
#define _O3index


namespace GElib{

  class O3index{
  public:

    int z;

    O3index(const int l, const int p): 
      z(2*l+(p==-1)){
      GELIB_ASSRT(p==1||p==-1);
    }

    O3index(const pair<int,int>& x):
      O3index(x.first,x.second){}

    constexpr O3index(int dummy): z(-1){} // change from int 


    int getl() const{
      return z/2;
    }

    int getp() const{
      return 1-2*(z%2);
    }

    bool operator==(const O3index& x) const{
      return z==x.z;
    }

    bool operator<(const O3index& x) const{
      return z<x.z;
    }

    bool operator<=(const O3index& x) const{
      return z<=x.z;
    }


    // ---- I/O ---------------------------------------------------------------------------------------------

    
    string str() const{
      return "("+to_string(z/2)+","+to_string(1-2*(z%2))+")";
    }

    friend ostream& operator<<(ostream& stream, const O3index& x){
      stream<<x.str(); return stream;
    }

  };

}


namespace std{
  template <>
  struct hash<GElib::O3index>{
    size_t operator()(const GElib::O3index& x) const {
      return hash<int>()(x.z);
    }
  };
}


#endif 
