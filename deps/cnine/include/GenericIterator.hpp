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


#ifndef _GenericIterator
#define _GenericIterator


namespace cnine{

  template<typename OWNER, typename OBJ>
  class GenericIterator{
  public:

    OWNER* owner;
    int i;

    GenericIterator(OWNER* _owner, const int _i=0): 
      owner(_owner), i(_i){}

    int operator++(){++i; return i;}

    int operator++(int a){++i; return i-1;}

    OBJ operator*() const{
      return (*owner)[i];
    }
      
    bool operator==(const GenericIterator& x) const{
      return i==x.i;
    }

    bool operator!=(const GenericIterator& x) const{
      return i!=x.i;
    }

  };


  template<typename OWNER, typename OBJ>
  class GenericConstIterator{
  public:

    const OWNER* owner;
    int i;

    GenericConstIterator(const OWNER* _owner, const int _i=0): 
      owner(_owner), i(_i){}

    int operator++(){++i; return i;}

    int operator++(int a){++i; return i-1;}

    OBJ operator*() const{
      return (*owner)[i];
    }
      
    bool operator==(const GenericConstIterator& x) const{
      return i==x.i;
    }

    bool operator!=(const GenericConstIterator& x) const{
      return i!=x.i;
    }

  };


}

#endif
