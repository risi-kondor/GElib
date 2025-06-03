/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
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
#include "ptr_triple_indexed_cache.hpp"

using namespace cnine;

class Widget: public observable<Widget>{
public:

  int v;

  Widget(const int _v): 
    observable(this),
    v(_v){}

};


int main(int argc, char** argv){
  cnine_session session(4);

  typedef ptr_triple_indexed_cache<Widget,Widget,Widget,int> CACHE;

  CACHE cache([](const Widget& x, const Widget& y, const Widget& z){return x.v+y.v+z.v;});

  Widget* w2=new Widget(2);
  Widget* w3=new Widget(3);
  Widget* w4=new Widget(4);

  int a=cache(w2,w3,w4);
  cout<<a<<endl;

  cout<<cache.size()<<endl;

  delete w2;

  cout<<cache.size()<<endl;

}
