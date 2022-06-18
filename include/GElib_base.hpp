
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

#include "Cnine_base.hpp"

using namespace std; 
//using namespace Cnine; 

#define _GELIB_VERSION "0.0.0 5/3/22"

#define GELIB_ASSERT(condition, message) if (!(condition)) {cout<<message<<endl; assert ((condition)); exit(-1); }
#define GELIB_UNIMPL() printf("GElib error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);
//#define GELIB_CPUONLY() if(dev!=0) {printf("GElib error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}
#define GELIB_ERROR(cmd) {CoutLock lk; cerr<<"GElib error in function '"<<__PRETTY_FUNCTION__<<"' : "<<cmd<<endl;} exit(1);

#define GELIB_CHECK(condition,err) if(!condition) {{cnine::CoutLock lk; cerr<<"GElib error in function '"<<__PRETTY_FUNCTION__<<"' : "<<err<<endl;} exit(1);};



namespace GElib{

  enum class SO3vec_format{parts,compact};

  inline int toint(const SO3vec_format& x){
    if(x==SO3vec_format::parts) return 0;
    //if(x==SO3vec_format::joint) return 1;
    if(x==SO3vec_format::compact) return 1;
    return 0;
  }

  // ---- Helpers --------------------------------------------------------------------------------------------

  inline ostream& operator<<(ostream& stream, const vector<int>& v){
    stream<<"(";
    for(int i=0; i<v.size()-1; i++)
      stream<<v[i]<<",";
    if(v.size()>0) stream<<v[v.size()-1]<<")";
    return stream;
  }


}


#define GENET_CHECK_NBU(a,b,cmd) if(a!=b) {{CoutLock lk; cerr<<"GEnet error in function "<<cmd<<": bundle dimensions do not match."<<endl;} exit(1);}
//#define GENET_CHECK_NBU2(a,b,c,cmd) if(a!=b||a!=c) {{CoutLock lk; cerr<<"GEnet error in function "<<cmd<<": bundle dimensions do not match."<<endl;} exit(1);}
//#define GENET_CHECK_NBU3(a,b,c,d,cmd) if(a!=b||a!=c||a!=d) {{CoutLock lk; cerr<<"GEnet error in function "<<cmd<<": bundle dimensions do not match."<<endl;} exit(1);}

#define GENET_CHECK_NBU2(a,b) if(a!=b) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": bundle dimensions do not match."<<endl;} exit(1);}
#define GENET_CHECK_NBU3(a,b,c) if(a!=b||a!=c) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": bundle dimensions do not match."<<endl;} exit(1);}
#define GENET_CHECK_NBU4(a,b,c,d) if(a!=b||a!=c||a!=d) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": bundle dimensions do not match."<<endl;} exit(1);}

#define GELIB_CHECK_NBU2(a,b) if(a!=b) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": bundle dimensions do not match."<<endl;} exit(1);}
#define GELIB_CHECK_NBU3(a,b,c) if(a!=b||a!=c) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": bundle dimensions do not match."<<endl;} exit(1);}
#define GELIB_CHECK_NBU4(a,b,c,d) if(a!=b||a!=c||a!=d) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": bundle dimensions do not match."<<endl;} exit(1);}



#define GELIB_CHECK_SO3VEC_FORMAT_IS_0(a) if(a!=0) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<"currently only works with 'SO3_format::parts'."<<endl;} exit(1);}

#define GELIB_CHECK_SO3FORMAT2(a,b) if(a!=b) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": SO3vec formats do not match."<<endl;} exit(1);}
#define GELIB_CHECK_SO3FORMAT3(a,b,c) if(a!=b||a!=c) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": SO3vec formats do not match."<<endl;} exit(1);}

#define GELIB_CHECK_TAU2(a,b) if(a!=b) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": SO3types do not match."<<endl;} exit(1);}
#define GELIB_CHECK_TAU3(a,b,c) if(a!=b||a!=c) {{CoutLock lk; cerr<<"GEnet error in function "<<__PRETTY_FUNCTION__<<": SO3types do not match."<<endl;} exit(1);}

#ifdef _GELIB_SO3CG_DEBUG
#define SO3CG_DEBUG(msg) ({{cnine::CoutLock lk; cerr<<msg<<endl;}})
#else 
#define SO3CG_DEBUG(msg) 
#endif 

// ---- CUDA STUFF ------------------------------------------------------------------------------------------


#define GELIB_CPUONLY() if(dev>0) {printf("Cengine error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}


#ifdef _WITH_CENGINE
#define GELIB_SO3PART_IMPL SO3partM
#define GELIB_SO3VEC_IMPL SO3vecM
#define GELIB_SO3PARTARRAY_IMPL SO3partArrayM
#define GELIB_SO3VECARRAY_IMPL SO3vecArrayM
#else 
#define GELIB_SO3PART_IMPL SO3partB
#define GELIB_SO3VEC_IMPL SO3vecB
#define GELIB_SO3PARTARRAY_IMPL SO3partArrayA
#define GELIB_SO3VECARRAY_IMPL SO3vecArrayA
#endif 


// ---- Conevenience functions --------------------------------------------------------------------------------

// move to cnine
namespace std{
template<>
struct hash<pair<int,int>>{
public:
  size_t operator()(const pair<int,int>& ix) const{
    return ((hash<int>()(ix.first)<<1)^hash<int>()(ix.second));
  }
};
}





#endif 
