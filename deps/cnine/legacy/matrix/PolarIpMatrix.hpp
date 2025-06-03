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

#ifndef _PolarIpMatrix
#define _PolarIpMatrix

#include "Cnine_base.hpp"
#include "RtensorA.hpp"
#include "array_pool.hpp"
#include "CSRmatrix.hpp"
#include "InterpolBilinear.hpp"


namespace cnine{

  template<typename TYPE>
  class PolarIpMatrix: public CSRmatrix<TYPE>{
  public:

    using CSRmatrix<TYPE>::arr;
    using CSRmatrix<TYPE>::arrg;
    using CSRmatrix<TYPE>::tail;
    using CSRmatrix<TYPE>::memsize;
    using CSRmatrix<TYPE>::dev;
    using CSRmatrix<TYPE>::is_view;
    using CSRmatrix<TYPE>::dir;
    using CSRmatrix<TYPE>::n;

    using CSRmatrix<TYPE>::reserve;
    using CSRmatrix<TYPE>::size;
    using CSRmatrix<TYPE>::offset;
    using CSRmatrix<TYPE>::size_of;
    using CSRmatrix<TYPE>::set_at;



    PolarIpMatrix(const int Nr, const int Nphi, const int N){

      RtensorA X(Gdims(Nr*Nphi,2));
      float x0=((float)(N-1.0))/2;
      for(int i=0; i<Nr; i++){
	float r=x0/Nr*((float)i+0.5);
	for(int j=0; j<Nphi; j++){
	  float phi=M_2_PI*((float)j)/((float)(Nphi));
	  X.set(i*Nphi+j,0,x0+r*cos(phi));
	  X.set(i*Nphi+j,1,x0+r*sin(phi));
	}
      }
      *this=InterpolBilinear<TYPE>(X,N,N);
    }


  public: // ---- Copying -------------------------------------------------------------------------------------


  public: // ---- Converions -------------------------------------------------------------------------------------


    PolarIpMatrix(const CSRmatrix<TYPE>& x):
      CSRmatrix<TYPE>(x){}


  };

}

#endif 

