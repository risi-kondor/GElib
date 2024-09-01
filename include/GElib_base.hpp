
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

#define GELIB_ASSRT(condition) \
  if(!(condition)) throw std::runtime_error("GElib error in "+string(__PRETTY_FUNCTION__)+" : failed assertion "+#condition+".");

#define GELIB_ASSERT(condition, message) if (!(condition)) {cout<<message<<endl; assert ((condition)); exit(-1); }

#define GELIB_CHECK(condition,err) if(!condition) {{cnine::CoutLock lk; cerr<<"GElib error in function '"<<__PRETTY_FUNCTION__<<"' : "<<err<<endl;} exit(1);};
#define GELIB_UNIMPL() printf("GElib error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);
//#define GELIB_CPUONLY() if(dev!=0) {printf("GElib error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}
#define GELIB_ERROR(msg) throw std::runtime_error("GElib error in "+string(__PRETTY_FUNCTION__)+" : "+msg+".");
//{cout<<33333<<endl; cnine::CoutLock lk; cout<<"GElib error in function '"<<__PRETTY_FUNCTION__<<"' : "<<cmd<<endl; exit(1);}

// ---- Copy, assign and convert warnings --------------------------------------------------------------------


#ifdef GELIB_COPY_WARNINGS
#define GELIB_COPY_WARNING() cout<<"\e[1mGElib:\e[0m "<<classname()<<" copied."<<endl;
#else 
#define GELIB_COPY_WARNING()
#endif 

#ifdef GELIB_MOVE_WARNINGS
#define GELIB_MOVE_WARNING() cout<<"\e[1mGElib:\e[0m "<<classname()<<" moved."<<endl;
#define GELIB_MCONVERT_WARNING(x) cout<<"\e[1mGElib:\e[0m "<<x.classname()<<" move converted to "<<classname()<<"."<<endl;
#else 
#define GELIB_MOVE_WARNING()
#define GELIB_MCONVERT_WARNING(x)
#endif 

#ifdef GELIB_ASSIGN_WARNINGS
#define GELIB_ASSIGN_WARNING() cout<<"\e[1mGElib:\e[0m "<<classname()<<" assigned."<<endl;
#define GELIB_MASSIGN_WARNING() cout<<"\e[1mGElib:\e[0m "<<classname()<<" move assigned."<<endl;
#else 
#define GELIB_ASSIGN_WARNING()
#define GELIB_MASSIGN_WARNING()
#endif 

#ifdef GELIB_CONVERT_WARNINGS
#define GELIB_CONVERT_WARNING(x) cout<<"\e[1mGElib:\e[0m "<<x.classname()<<" converted to "<<classname()<<"."<<endl;
#else 
#define GELIB_CONVERT_WARNING(x)
#endif 


// --------------------------------------------------------------------------------------------------


#ifdef GELIB_RANGE_CHECKING
#define GELIB_CHECK_RANGE(expr) expr
#else 
#define GELIB_CHECK_RANGE(expr)
#endif 

namespace GElib{

  enum class SO3vec_format{parts,compact};

  inline int toint(const SO3vec_format& x){
    if(x==SO3vec_format::parts) return 0;
    //if(x==SO3vec_format::joint) return 1;
    if(x==SO3vec_format::compact) return 1;
    return 0;
  }

  // ---- Helpers --------------------------------------------------------------------------------------------


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


// should go into cnine_base.hpp
inline ostream& operator<<(ostream& stream, const pair<int,int>& x){
  stream<<"("<<x.first<<","<<x.second<<")";
  return stream;
}





#endif 
