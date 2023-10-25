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

using namespace cnine;
using namespace GElib;

const int maxl1=2;
const int maxl=4;


int main(int argc, char** arg){

  ofstream ofs("SO3part_addCGproduct_explicit_calls.inc");
  
  ofs<<"  switch(l1){\n";
  for(int l1=0; l1<=maxl1; l1++){
    ofs<<"      case "<<l1<<":\n";

    ofs<<"        switch(l2){\n";
    for(int l2=0; l2<=maxl1; l2++){
      ofs<<"        case "<<l2<<":\n";

      ofs<<"          switch(l){\n";
      for(int l=std::abs(l1-l2); l<=l1+l2 && l<=maxl; l++){
	ofs<<"          case "<<l<<": ";
	ofs<<"SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_"<<l1<<"_"<<l2<<"_"<<l<<">"
	   <<"<<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;"<<endl;
      }
      ofs<<"          }"<<endl;
      ofs<<"        break;"<<endl;
    }
    ofs<<"        }"<<endl<<endl;
    ofs<<"      break;"<<endl;

  }
  ofs<<"      }"<<endl;

  ofs.close();
}
