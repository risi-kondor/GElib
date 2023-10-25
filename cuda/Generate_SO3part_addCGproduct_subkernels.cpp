/*
 * This file is part of GElib, a C++/CUDA library for group
 * equivariant tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include <fstream>

extern GElib::SO3_CGbank SO3_cgbank;

using namespace cnine;
using namespace GElib;

const int maxl1=2;
const int maxl=4;


int main(int argc, char** arg){

  ofstream ofs("SO3part_addCGproduct_subkernels.inc");
  
  for(int l1=0; l1<=maxl1; l1++){

    for(int l2=0; l2<=maxl1; l2++){

      for(int l=std::abs(l1-l2); l<=l1+l2 && l<=maxl; l++){
	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));

	ofs<<"__forceinline__ __device__ void SO3part_addCGproduct_explicit_kernel_"<<l1<<"_"<<l2<<"_"<<l<<
	  "(const float* xpr, const float* xpi, const float* ypr, const float* ypi, const int ys, float* rpr, float* rpi, const int rs){"<<endl;

	for(int m=-l; m<=l; m++){

	  ofs<<"  rpr["<<m+l<<"*rs]+="<<endl;
	  for(int m1=max(-l1,m-l2); m1<=min(l1,m+l2); m1++){
	    int m2=m-m1;
	    float c=C(m1+l1,m2+l2); 
	    string cs=to_string(c);
	    //if(c==floor(c)) cs=cs+".";
	    //c=1.0;
	    ofs<<"    ("<<cs<<"f)*(xpr["<<m1+l1<<"]*ypr["<<m2+l2<<"*ys]-xpi["<<m1+l1<<"]*ypi["<<m2+l2<<"*ys])";
	    if(m1<min(l1,m+l2)) ofs<<"+"<<endl;
	  }
	  ofs<<";"<<endl;

	  ofs<<"  rpi["<<m+l<<"*rs]+="<<endl;
	  for(int m1=max(-l1,m-l2); m1<=min(l1,m+l2); m1++){
	    int m2=m-m1;
	    float c=C(m1+l1,m2+l2); 
	    string cs=to_string(c);
	    //if(c==floor(c)) cs=cs+".";
	    //c=1.0;
	    ofs<<"    ("<<cs<<"f)*(xpr["<<m1+l1<<"]*ypi["<<m2+l2<<"*ys]+xpi["<<m1+l1<<"]*ypr["<<m2+l2<<"*ys])";
	    if(m1<min(l1,m+l2)) ofs<<"+"<<endl;
	  }
	  ofs<<";"<<endl;

	}

	ofs<<"}"<<endl<<endl;

      }
    }
  }

  ofs.close();


}
