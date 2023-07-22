// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3
#define _SO3

#include "object_bank.hpp"
#include "Tensor.hpp"
#include "GprodSpaceBank.hpp"
#include "SO3CouplingMatrices.hpp"


namespace GElib{


  class SO3{
  public:

    typedef int IrrepIx;
  
    static SO3CouplingMatrices coupling_matrices;
    static GprodSpaceBank<SO3> product_space_bank;


  public: // ---- CG-products --------------------------------------------------------------------------------


    int CGmultiplicity(const int l1, const int l2, const int l){
      return (l>=abs(l1-l2))&&(l<=l1+l2);
    }

    static void for_each_CGcomponent(const int l1, const int l2, 
      const std::function<void(const int&, const int)>& lambda){
      for(int l=std::abs(l1-l2); l<=l1+l2; l++)
	lambda(l,1);
    }


  public: // ---- Coupling coefficients ---------------------------------------------------------------------


    static cnine::TensorView<double> coupling(int j1, int j2, int j3, int j){
      return coupling_matrices(j1,j2,j3,j).view();
    }


  public: // ---- SnSpaces ----------------------------------------------------------------------------------


    static GprodSpaceObj<SO3>* space(const int& ix){
      return product_space_bank(ix);
    }

    static GprodSpaceObj<SO3>* space(GprodSpaceObj<SO3>* x, GprodSpaceObj<SO3>* y){
      return product_space_bank(x,y);
    }

    static GprodSpaceObj<SO3>* FmoveL(GprodSpaceObj<SO3>* x){
      GELIB_ASSRT(x->left!=nullptr && x->right!=nullptr);
      GELIB_ASSRT(x->left->left!=nullptr && x->left->right!=nullptr);
      auto t=product_space_bank(x->left->right,x->right);
      return product_space_bank(x->left->left,t);
    }

    static GprodSpaceObj<SO3>* FmoveR(GprodSpaceObj<SO3>* x){
      GELIB_ASSRT(x->left!=nullptr && x->right!=nullptr);
      GELIB_ASSRT(x->right->left!=nullptr && x->right->right!=nullptr);
      auto t=product_space_bank(x->left,x->right->left);
      return product_space_bank(t,x->right->right);
    }

    /*
    static HomMap<SO3,double> coupling(GprodSpaceObj<SO3>* x, GprodSpaceObj<SO3>* y){
      GELIB_ASSRT(x->is_isomorphic(*y));
      //HomMap<SO3,double> Tx=x->standardizer(); //ccoupling(x);
      //HomMap<SO3,double> Ty=ccoupling(y);
      //return Ty*(cnine::transp(Tx));
      HomMap<SO3,double> R=HomMap<SO3,double>::zero(*x,*y);
      return R;
    }
    */

    /*
    static HomMap<SO3,double> lcoupling(GprodSpaceObj<SO3>* x){
      HomMap<SO3,double> Z=HomMap<SO3,double>::zero(*x,*x);
      //if(x->is_sequential()) return R;
      //if(x->left)
      if(x->is_leaf()) return Z;
      auto R=rcoupling(x->right);
      GprodSpaceObj* u=new GprodSpaceObj()

      return Z;
    }
    */

    //static left_canonical
    

  public: // ---- I/O ---------------------------------------------------------------------------------------

    static string repr(){
      return "SO3";
    }

  };

}

#endif 


    /*
    static void init(){
      SO3::coupling_matrices=cnine::object_bank<CouplingSignature,cnine::Tensor<float> >
	([](const CouplingSignature& x){
	  auto R=new cnine::Tensor<float>({x.j1+x.j2+1-std::abs(x.j1-x.j2),x.j2+x.j3+1-std::abs(x.j2-x.j3)},cnine::fill_zero());
	  auto deltasq=[](const int a, const int b, const int c){return cnine::delta_factor.squared(a,b,c);};
	  cnine::DeltaFactor& delta=cnine::delta_factor;
	  int j1=x.j1;
	  int j2=x.j2;
	  int j3=x.j3;
	  int j=x.j;

	  int offs1=std::abs(x.j1-x.j2);
	  vector<cnine::frational> d1(x.j1+x.j2+1-std::abs(x.j1-x.j2));
	  //for(int ja=std::abs(x.j1-x.j2); ja<=x.j1+x.j2; ja++)
	  //d1[ja-offs1]=delta.squared(j1,j2,ja)*delta.squared(j1,j2,b);

	  for(int ja=std::abs(x.j1-x.j2); ja<=x.j1+x.j2; ja++){
	    for(int jb=std::abs(x.j2-x.j3); jb<=x.j2+x.j3; jb++){
	    }
	  }

	  return R;
	});
    }
    */
    //static cnine::object_bank<CouplingSignature,cnine::Tensor<float> > coupling_matrices;
    /*
    Gtype<SO3> product(const Gtype<SO3>& x, Gtype<SO3>& y){
      Gtype<SO3> R;
      for(auto& p:x)
	for(auto& q:y)
	  for(int l=std::abs(x.first-y.first); l<=x.first+y.first; l++)
	    R[l]+=x.second*y.second;
      return R;
    }
    */

