// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _RootedBinaryTree
#define _RootedBinaryTree

namespace GElib{
  
  template<typename LABEL>
  class RootedBinaryTree{
  public:
    
    LABEL label;
    RootedBinaryTree* left=nullptr;
    RootedBinaryTree* right=nullptr;
    
    ~RootedBinaryTree(){
      delete left;
      delete right;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    RootedBinaryTree(const LABEL _label):
      label(_label){}

    RootedBinaryTree(RootedBinaryTree* _left, RootedBiaryTree* _right):
      left(_left), right(_right){}


  public: // ---- I/O ---------------------------------------------------------------------------------------
    
    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<label<<endl;
      if(left) oss<<left.str(indent+"  ");
      if(right) oss<<right.str(indent+"  ");
      return oss.str();
    }

  };

}

#endif 
