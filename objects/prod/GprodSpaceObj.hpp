// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GprodSpaceObj
#define _GprodSpaceObj

#include "EndMap.hpp"
#include "Gtype.hpp"
#include "GprodIsotypic.hpp"
#include "quintuple_map.hpp"


namespace GElib{

  template<typename GROUP>
  class GprodSpaceObj{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef GprodIsotypic<GROUP> _Isotypic;

    int id=0;
    int nnodes=1;
    _IrrepIx irrep;
    GprodSpaceObj* left=nullptr;
    GprodSpaceObj* right=nullptr;
    map<_IrrepIx,_Isotypic> isotypics;
    Gtype<GROUP> tau;

    cnine::quintuple_map<_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,int>* Ltriple_index_map=nullptr;
    cnine::quintuple_map<_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,int>* Rtriple_index_map=nullptr;

    EndMap<GROUP,double>* _standardizer=nullptr;
    EndMap<GROUP,double>* _rstandardizer=nullptr;

    EndMap<GROUP,double>* _Lmove_map=nullptr;

    ~GprodSpaceObj(){
      delete Ltriple_index_map;
      delete Rtriple_index_map;
      delete _standardizer;
      delete _rstandardizer;
      delete _Lmove_map;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    GprodSpaceObj(_IrrepIx _irrep, const int _id): 
      id(_id), irrep(_irrep), tau(_irrep){
      isotypics[_irrep]=_Isotypic(_irrep,1);
    }

    GprodSpaceObj(GprodSpaceObj* _x, GprodSpaceObj* _y, const int _id): 
      id(_id), nnodes(_x->nnodes+_y->nnodes+1), left(_x), right(_y), tau(tprod(_x->tau,_y->tau)){
      for(auto x:_x->isotypics)
	for(auto y:_y->isotypics)
	  GROUP::for_each_CGcomponent(x.second.ix,y.second.ix,[&](const _IrrepIx& _irrep, const int n){
	  auto it=isotypics.find(_irrep);
	  if(it!=isotypics.end()) it->second.n+=n*x.second.n*y.second.n;
	  else isotypics[_irrep]=_Isotypic(_irrep,n);
	});
      make_offsets();
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    GprodSpaceObj(const GprodSpaceObj& x)=delete;


  public: // ---- Access ------------------------------------------------------------------------------------


    Gtype<GROUP> get_tau() const{
      return tau;
    }

    bool is_leaf() const{
      return (left==nullptr);
    }

    bool is_standard(){
      if(is_leaf()) return true;
      if(!right->is_leaf()) return false;
      return left->is_standard();
    }

    bool is_reverse(){
      if(is_leaf()) return true;
      if(!left->is_leaf()) return false;
      return right->is_reverse();
    }

    bool is_isomorphic(const GprodSpaceObj& y) const{
      unordered_map<_IrrepIx,int> xcounts;
      unordered_map<_IrrepIx,int> ycounts;
      add_counts(xcounts);
      y.add_counts(ycounts);
      return xcounts==ycounts;
    }

    //GprodIndex<GROUP> index(const _IrrepIx& l, const int n){
    //GELIB_ASSRT(n<tau[l]);
    //auto 
    //}

    //int index(const GprodIndex& x){
    //GELIB_ASSRT(x.size()==nnodes);
    //}





    int Ltriple_index(const int l1, const int l2, const int l3, const int l12, const int l){
      if(!Ltriple_index_map) make_Ltriple_index_map();
      return Ltriple_index_map[{l1,l2,l3,l12,l}];
    }

    int Rtriple_index(const int l1, const int l2, const int l3, const int l23, const int l){
      if(!Rtriple_index_map) make_Rtriple_index_map();
      return Rtriple_index_map[{l1,l2,l3,l23,l}];
    }

    const EndMap<GROUP,double>& standardizer(){
      if(!_standardizer) make_standardizer();
      return *_standardizer;
    }

    const EndMap<GROUP,double>& reverse_standardizer(){
      if(!_rstandardizer) make_rstandardizer();
      return *_rstandardizer;
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    GprodSpaceObj* shiftL() const{
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      GELIB_ASSRT(right->left!=nullptr && right->right!=nullptr);
      auto t=GROUP::space(left,right->left);
      return GROUP::space(t,right->right);
    }

    GprodSpaceObj* shiftR() const{
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      GELIB_ASSRT(left->left!=nullptr && left->right!=nullptr);
      auto t=GROUP::space(left->right,right);
      return GROUP::space(left->left,t);
    }

    GprodSpaceObj* cleave() const{
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      GprodSpaceObj* u=left->standardize();
      GprodSpaceObj* v=right->reverse_standardize();
      return GROUP::space(u,v);
    }

    GprodSpaceObj* standardize() const{
      if(is_standard()) return;
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      GprodSpaceObj* u=cleave();
      while(u->right->is_leaf()){
	u=u->shiftL();
      }
    }

    GprodSpaceObj* reverse_standardize() const{
      if(is_reverse()) return;
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      GprodSpaceObj* u=cleave();
      while(u->left->is_leaf()){
	u=u->shiftR();
      }
    }

    

    GprodSpaceObj* Lmove() const{
      return GROUP::FmoveL(this);
    }

    GprodSpaceObj* Rmove() const{
      return GROUP::FmoveR(this);
    }

    void make_standardizer(){
      if(_standardizer) return;
      if(is_standard()){
	_standardizer=new EndMap<GROUP,double>(tau,cnine::fill_identity());
	return;
      }
    }

    void make_rstandardizer(){
    }

    const EndMap<GROUP,double>& LmoveMap(){
      if(!_Lmove_map) make_Lmove_map();
      return *_Lmove_map;
    }


  private: // ---- Internal ----------------------------------------------------------------------------------


    void add_counts(unordered_map<_IrrepIx,int>& counts) const{
      if(is_leaf()){
	counts[irrep]++;
	return;
      }
      left->add_counts(counts);
      right->add_counts(counts);
    }

    void make_offsets(){
      if(is_leaf()) return;

      cnine::Llist<_IrrepIx> llabels; 
      for(auto& p:left->isotypics)
	llabels.insert(p.first);

      cnine::Llist<_IrrepIx> rlabels; 
      for(auto& p:right->isotypics)
	rlabels.insert(p.first);

      for(auto& p:isotypics)
	p.second.offsets=
	  new cnine::Lmatrix<_IrrepIx,_IrrepIx,int>(llabels,rlabels,cnine::fill_constant<int>(-1));

      Gtype<GROUP> offs;
      for(auto& p:left->isotypics)
	for(auto& q:right->isotypics)
	  GROUP::for_each_CGcomponent(p.second.ix,q.second.ix,[&](const _IrrepIx& l, const int n){
	      isotypics[l].offsets->set(p.second.ix,q.second.ix,offs[l]);
	      offs[l]+=n*p.second.n*q.second.n;
	});

    }

    void make_Ltriple_index_map(){
      GprodSpaceObj& x=*left->left;
      GprodSpaceObj& y=*left->right;
      GprodSpaceObj& z=*right;

      Gtype<GROUP> offs;
      for(auto& p1:x.tau){
	auto l1=p1.first;
	for(auto& p2:y.tau){
	  auto l2=p2.first;
	  GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l12, const int m12){
	      for(auto& p3:z.tau){
		auto l3=p3.first;
		GROUP::for_each_CGcomponent(l12,l3,[&](const _IrrepIx& l, const int m){
		    Ltriple_index_map({l1,l2,l3,l12,l})=offs[l];
		    offs[l]+=m12*m*p1.second*p2.second*p3.second;
		  });
	      }
	    });
	}
      }
    }
    
    void make_Rtriple_index_map(){
      GprodSpaceObj& x=*left->left;
      GprodSpaceObj& y=*left->right;
      GprodSpaceObj& z=*right;

      Gtype<GROUP> offs;
      for(auto& p1:x.tau){
	auto l1=p1.first;
	for(auto& p2:y.tau){
	  auto l2=p2.first;
	  for(auto& p3:z.tau){
	    auto l3=p3.first;
	    GROUP::for_each_CGcomponent(l2,l3,[&](const _IrrepIx& l23, const int m23){
		GROUP::for_each_CGcomponent(l1,l23,[&](const _IrrepIx& l, const int m){
		    Rtriple_index_map({l1,l2,l3,l23,l})=offs[l];
		    offs[l]+=m23*m*p1.second*p2.second*p3.second;
		  });
	      });
	  }
	}
      }
    }


    void make_Lmove_map(){
      _Lmove_map=new EndMap<GROUP,double>(tau,cnine::fill_zero());
      GprodSpaceObj& x=*left->left;
      GprodSpaceObj& y=*left->right;
      GprodSpaceObj& z=*right;

      for(auto& p1:x.tau){
	auto l1=p1.first;
	for(auto& p2:y.tau){
	  auto l2=p2.first;
	  for(auto& p3:z.tau){
	    auto l3=p3.first;
	    int m0=p1.second*p2.second*p3.second;
	    
	    GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l12, const int m12){
		GROUP::for_each_CGcomponent(l12,l3,[&](const _IrrepIx& l, const int mL){
		    GROUP::for_each_CGcomponent(l2,l3,[&](const _IrrepIx& l23, const int m23){
			GROUP::for_each_CGcomponent(l1,l23,[&](const _IrrepIx& ld, const int mR){
			    if(l==ld){
			      int loffs=Ltriple_index(l1,l2,l3,l12,l);
			      int roffs=Rtriple_index(l1,l2,l3,l23,l);
			      int lwidth=m0*m12*mL;
			      int rwidth=m0*m23*mR;
			      auto T=cnine::Tensor<double>::constant({lwidth,rwidth},GROUP::coupling(l1,l2,l3,l,l12,l23));
			      _Lmove_map->map[l].block({loffs,roffs},{lwidth,rwidth})=T;
			    }
			  });
		      });
		  });
	      });
	  }
	}
      }
    }
    

  public: // ---- I/O ---------------------------------------------------------------------------------------


    string reprr() const{
      ostringstream oss;
      if(!left) oss<<"("<<irrep<<")";
      else oss<<"("<<left->reprr()<<"*"<<right->reprr()<<")";
      return oss.str();
    }

    string repr() const{
      ostringstream oss;
      oss<<"Gspace<"<<GROUP::repr()<<">"<<reprr();
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<endl;
      for(auto p:isotypics)
	oss<<p.second.str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GprodSpaceObj& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 


    /*
    for_each_Ltriple(std::function<void(const _IrrepIx&, const _IrrepIx&, const _IrrepIx&, const _IrrepIx&, const _IrrepIx&, const int, const int)>){
      GprodSpaceObj& x=*left->left;
      GprodSpaceObj& y=*left->right;
      GprodSpaceObj& z=*right;

      Gtype<GROUP> offs;
      for(auto& p1:x.tau){
	auto l1=p1.first;
	for(auto& p2:y.tau){
	  auto l2=p2.first;
	  for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l12, const int m12){
	      for(auto& p3:z.tau){
		auto l3=p3.first;
		for_each_CGcomponent(l12,l3,[&](const _IrrepIx& l, const int m){
		    int M=m12*m*p1.second*p2.second*p3.second;
		    lambda(l1,l2,l3,l12,l,coffs[l],M);
		    coffs[l]+=M;
		  });
	      }
	    });
	}
	}
    */
      /*
      for(auto x:_x->isotypics)
	for(auto y:_y->isotypics)
	  GROUP::for_each_CGcomponent(x.second->ix,y.second->ix,[&](const _IrrepIx& _irrep, const int m){
	  auto it=isotypics.find(_irrep);
	  if(it!=isotypics.end()) it->second->m+=m*x.second->m*y.second->m;
	  else isotypics[_irrep]=new _Isotypic(_irrep,m);
	});
      */
      //isotypics[_irrep]=new _Isotypic(_irrep);
      //for(auto p:isotypics)
      //oss<<indent<<"  "<<*p.second<<endl;
    //map<_IrrepIx,_Isotypic*> isotypics;
    //typedef Gisotypic<GROUP> _Isotypic;
