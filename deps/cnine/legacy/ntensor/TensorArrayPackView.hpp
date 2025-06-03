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

#ifndef _CnineTensorArrayPackView
#define _CnineTensorArrayPackView

#include "GElib_base.hpp"
#include "TensorPackView.hpp"
#include "TensorArrayView.hpp"


namespace cnine{

  template<typename TYPE>
  class TensorArrayPackView: public cnine::TensorPackView<TYPE>{
  public:

    typedef TensorPackView<TYPE> TensorPackView;

    using TensorPackView::TensorPackView;
    using TensorPackView::dims;
    using TensorPackView::strides;
    using TensorPackView::arr;
    using TensorPackView::size;
    using TensorPackView::offset;

    int ak=0;

  public: // ---- Constructors --------------------------------------------------------------------------------

  public: // ---- Access --------------------------------------------------------------------------------------



  public: // individual tensor arrays


    int nadims(const int i) const{
      return ak;
    }

    int nddims(const int i) const{
      return dims(i).size()-ak;
    }

    Gdims adims(const int i) const{
      return dims(i).chunk(0,ak);
    }

    Gdims ddims(const int i) const{
      return dims(i).chunk(ak);
    }

    Gdims astrides(const int i) const{
      return strides(i).chunk(0,ak);
    }

    Gdims dstrides(const int i) const{
      return strides(i).chunk(ak);
    }

    TensorArrayView<TYPE> operator[](const int i) const{
      return TensorArrayView<TYPE>(arr+offset(i),ak,adims(i),strides(i));
    }




  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayPackView";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Array "<<i<<":"<<endl;
	oss<<(*this)[i].str(indent+"  ")<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const TensorArrayPackView& v){
      stream<<v.str(); return stream;}


    

  };

}

#endif
