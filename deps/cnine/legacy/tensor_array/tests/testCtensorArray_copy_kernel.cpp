/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "CtensorObj_funs.hpp"
#include "CtensorArray_funs.hpp"

#include "CtensorA_copy_cop.hpp"

#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "CellwiseBinaryCmap.hpp"
#include "MVprodCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;

typedef CtensorA_copy_cop Ctensor_copy;


int main(int argc, char** argv){
  cnine_session genet;
  cout<<endl;

  CtensorArray A(dims(2),dims(2,2),[](const Gindex& aix, const Gindex& cix){return aix(0);});

  //printl("cellwise",cellwise<Ctensor_copy>(A));

}
