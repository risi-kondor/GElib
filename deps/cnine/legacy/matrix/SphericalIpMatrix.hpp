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

#ifndef _SphericalIpMatrix
#define _SphericalIpMatrix

#include "Cnine_base.hpp"
#include "RtensorA.hpp"
#include "array_pool.hpp"
#include "CSRmatrix.hpp"
#include "InterpolTrilinear.hpp"


namespace cnine{

  template<typename TYPE>
  class SphericalIpMatrix: public CSRmatrix<TYPE>{
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



    SphericalIpMatrix(const int Nr, const int Ntheta, const int Nphi, const int N){

      RtensorA X(Gdims(Nr*Ntheta*Nphi,3));
      float x0=((float)(N-1.0))/2;
      for(int i=0; i<Nr; i++){
	float r=x0/Nr*((float)i+0.5);
	for(int j=0; j<Nphi; j++){
	  float phi=M_2_PI*((float)j)/((float)(Nphi));
	  for(int k=0; k<Ntheta; k++){
	    float theta=M_PI*(((float)(k)+0.5)/Ntheta);
	    X.set((i*Ntheta+k)*Nphi+j,0,x0+r*cos(phi)*sin(theta));
	    X.set((i*Ntheta+k)*Nphi+j,1,x0+r*sin(phi)*sin(theta));
	    X.set((i*Ntheta+k)*Nphi+j,2,x0+r*cos(theta));
	  }
	}
      }
      *this=InterpolTrilinear<TYPE>(X,N,N,N);
    }


  public: // ---- Copying -------------------------------------------------------------------------------------


  public: // ---- Converions -------------------------------------------------------------------------------------


    SphericalIpMatrix(const CSRmatrix<TYPE>& x):
      CSRmatrix<TYPE>(x){}


  };

}

#endif 

