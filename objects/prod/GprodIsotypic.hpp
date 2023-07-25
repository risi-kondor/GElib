// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GprodIsotypic
#define _GprodIsotypic

#include "Lmatrix.hpp"
#include "cachedf.hpp"
#include "SnBasis.hpp"


namespace GElib{

  template<typename GROUP>
  class CGprodBasisObj;


  template<typename GROUP>
  class GprodIsotypic{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef cnine::Lmatrix<_IrrepIx,_IrrepIx,int> _Lmatrix;


    _IrrepIx ix;
    int n=0;
    CGprodBasisObj<GROUP>* owner;

    //_Lmatrix* offsets=nullptr; // currently unused
    
    ~GprodIsotypic(){
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    GprodIsotypic(){}

    GprodIsotypic(CGprodBasisObj<GROUP>* _owner, const _IrrepIx& _ix, const int _n=0):
      owner(_owner),ix(_ix), n(_n){cout<<" "<<ix<<endl;}

    //GprodIsotypic(const _IrrepIx& _ix, const cnine::Llist<_IrrepIx>& _llabels, const cnine::Llist<_IrrepIx>& _rlabels):
    //ix(_ix), offsets(new _Lmatrix(_llabels,_rlabels,cnine::fill_constant<int>(-1))){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    GprodIsotypic(const GprodIsotypic& x)=delete;


  public: // ---- Sn-action ---------------------------------------------------------------------------------


    cnine::Tensor<double>& standardizing_map(){
      return owner->standardizing_map().maps[ix];
    }

    /*
    cnine::cachedF<cnine::Tensor<double> > Sn_map=
      cnine::cachedF<cnine::Tensor<double> >([&](){
	  if(!owner->is_standard())
	    return cnine::transp(standardizing_map())*owner->standard_form().isotypics[ix].Sn_map()*standardizing_map();
	  if(owner->is_leaf()) return cnine::Tensor<double>({n,n},cnine::fill_identity());
	  if(owner->is_stem()) return cnine::Tensor<double>({n,n},cnine::fill_identity());
	  //auto M=owner->transpose_last_map().maps[ix];
	  //BlockDiagonalize blocke
	  //if(u.left->is_leaf()){
	  //}
	  return cnine::Tensor<double>();
	});
    */

    cnine::cachedF<SnBasis<double> > Sn_basis=
      cnine::cachedF<SnBasis<double> >([&](){
	  if(!owner->is_standard()){cout<<"a"<<endl;
	    return owner->standard_form().isotypics[ix]->Sn_basis().conjugate(standardizing_map());}
	  if(owner->is_leaf()){cout<<"b"<<endl;
	    return SnBasis<double>( {{{1},n}}, cnine::Tensor<double>({n,n},cnine::fill_identity()));}
	  if(owner->is_stem()){cout<<"c"<<endl;
	    int parity=GROUP::CG_sign_rule(owner->left->irrep,owner->right->irrep,ix,0); //TODO 
	    if(parity==1) return SnBasis<double>({{{2},n}},cnine::Tensor<double>({n,n},cnine::fill_identity()));
	    else return SnBasis<double>({{{1,1},n}},cnine::Tensor<double>({n,n},cnine::fill_identity()));
	  }
	  cout<<"d"<<endl;
	  auto M=owner->transpose_last_map().maps[ix];
	  cout<<"diaginalizing:"<<endl<<M<<endl;
	  return SnBasis<double>({{{2,1},3}},cnine::Tensor<double>::sequential({3,3}));
	});


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string repr() const{
      ostringstream oss;
      oss<<"Isotypic<"<<GROUP::repr()<<">("<<ix<<","<<n<<")";
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent+repr()<<endl;
      //if(offsets) oss<<offsets->str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GprodIsotypic& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
