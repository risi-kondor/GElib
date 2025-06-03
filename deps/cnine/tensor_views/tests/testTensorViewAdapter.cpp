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

#include "CnineSession.hpp"
#include "RtensorA.hpp"

#include "GindexFuser.hpp"
#include "GindexSymm.hpp"
#include "IndexingAdapter.hpp"


using namespace cnine;


int main(int argc, char** argv){

  cnine_session session(4);

  //GindexFuser fuse0(3,3);
  GindexSymm fuse0(3,3);
  cout<<fuse0(2,2)<<endl;

  RtensorA A({9},fill::sequential);
  cout<<A<<endl;


  IndexingAdapter1<RtensorA,GindexSymm> B(A,fuse0);
  //cout<<B(0,2)<<endl;
  cout<<B.to<RtensorA>()<<endl;

}
