// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _CGprodBasisObj
#define _CGprodBasisObj

#include "EndMap.hpp"
#include "Gtype.hpp"
#include "CGprodBasisIsotypic.hpp"
#include "cachedf.hpp"
#include "triple_map.hpp"
#include "quintuple_map.hpp"


namespace GElib{


  template<typename GROUP>
  class CGprodBasisObj{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef CGprodBasisIsotypic<GROUP> _Isotypic;

    int id=0;
    int nnodes=1;
    _IrrepIx irrep;
    CGprodBasisObj* left=nullptr;
    CGprodBasisObj* right=nullptr;
    map<_IrrepIx,_Isotypic*> isotypics;
    Gtype<GROUP> tau;

    ~CGprodBasisObj(){
      for(auto p:isotypics) delete p.second;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    CGprodBasisObj(_IrrepIx _irrep, const int _id): 
      id(_id), irrep(_irrep), tau(_irrep){
      isotypics[_irrep]=new _Isotypic(this,_irrep,1);
    }

    CGprodBasisObj(CGprodBasisObj* _x, CGprodBasisObj* _y, const int _id): 
      id(_id), nnodes(_x->nnodes+_y->nnodes+1), left(_x), right(_y), tau(tprod(_x->tau,_y->tau)){
      for(auto& x:_x->isotypics)
	for(auto& y:_y->isotypics)
	  GROUP::for_each_CGcomponent(x.second->ix,y.second->ix,[&](const _IrrepIx& _irrep, const int n){
	      auto it=isotypics.find(_irrep);
	      if(it!=isotypics.end()) it->second->n+=n*x.second->n*y.second->n;
	      else isotypics[_irrep]=new _Isotypic(this,_irrep,n*x.second->n*y.second->n);
	    });
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    CGprodBasisObj(const CGprodBasisObj& x)=delete;


  public: // ---- Boolean -----------------------------------------------------------------------------------


    bool is_leaf() const{
      return (left==nullptr);
    }

    bool is_stem() const{
      if(left==nullptr) return false;
      return left->is_leaf() && right->is_leaf();
    }

    bool is_standard() const{
      if(is_leaf()) return true;
      if(!right->is_leaf()) return false;
      return left->is_standard();
    }

    bool is_reverse() const{
      if(is_leaf()) return true;
      if(!left->is_leaf()) return false;
      return right->is_reverse();
    }

    bool is_isomorphic(const CGprodBasisObj& y) const{
      unordered_map<_IrrepIx,int> xcounts;
      unordered_map<_IrrepIx,int> ycounts;
      add_counts(xcounts);
      y.add_counts(ycounts);
      return xcounts==ycounts;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    Gtype<GROUP> get_tau() const{
      return tau;
    }

    CGprodBasisIsotypic<GROUP>& isotypic(const _IrrepIx& ix){
      GELIB_ASSRT(isotypics.find(ix)!=isotypics.end());
      return *isotypics[ix];
    }

    void for_each_leaf(std::function<void(CGprodBasisObj*)> lambda){
      if(is_leaf()) lambda(this);
      else{
	left->for_each_leaf(lambda);
	right->for_each_leaf(lambda);
      }
    }

    void for_each_leaf_reverse(std::function<void(CGprodBasisObj*)> lambda){
      if(is_leaf()) lambda(this);
      else{
	right->for_each_leaf(lambda);
	left->for_each_leaf(lambda);
      }
    }

    void for_each_block(std::function<void(const _IrrepIx& l1, const _IrrepIx& l2, const _IrrepIx& l, int offs, int n)> lambda){
      GELIB_ASSRT(!is_leaf());
      CGprodBasisObj& x=*left;
      CGprodBasisObj& y=*right;

      for(auto& p1:x.tau){
	auto l1=p1.first;
	auto n1=p1.second;
	for(auto& p2:y.tau){
	  auto l2=p2.first;
	  auto n2=p2.second;
	  GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l, const int m){
	      lambda(l1,l2,l,offset_map()(l1,l2,l),m*n1*n2);
	    });
	}
      }
    }

    void for_each_subisotypic_pair(const int _l, std::function<void(_Isotypic& I1, _Isotypic& I2, int offs, int n)> lambda){
      GELIB_ASSRT(!is_leaf());
      CGprodBasisObj& x=*left;
      CGprodBasisObj& y=*right;

      for(auto& p1:x.tau){
	auto l1=p1.first;
	for(auto& p2:y.tau){
	  auto l2=p2.first;
	  GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l, const int m){
	      if(l==_l) lambda(*x.isotypics[l1],*y.isotypics[l2],offset_map()(l1,l2,l),m*p1.second*p2.second);
	    });
	}
      }
    }


  public: // ---- Other bases -----------------------------------------------------------------------------------


    cnine::cachedf<CGprodBasisObj> shift_left=
      cnine::cachedf<CGprodBasisObj>([&](){
	  GELIB_ASSRT(left!=nullptr && right!=nullptr);
	  GELIB_ASSRT(right->left!=nullptr && right->right!=nullptr);
	  auto t=GROUP::space(left,right->left);
	  return GROUP::space(t,right->right);
	});

    cnine::cachedf<CGprodBasisObj> shift_right=
      cnine::cachedf<CGprodBasisObj>([&](){
	  GELIB_ASSRT(left!=nullptr && right!=nullptr);
	  GELIB_ASSRT(left->left!=nullptr && left->right!=nullptr);
	  auto t=GROUP::space(left->right,right);
	  return GROUP::space(left->left,t);
	});

    cnine::cachedf<CGprodBasisObj> swap=
      cnine::cachedf<CGprodBasisObj>([&](){
	  GELIB_ASSRT(left!=nullptr && right!=nullptr);
	  return GROUP::space(right,left);
	});

    cnine::cachedf<CGprodBasisObj> standard_form=
      cnine::cachedf<CGprodBasisObj>([&](){
	  CGprodBasisObj* u; 
	  bool first=true;
	  for_each_leaf([&](CGprodBasisObj* x){
	      if(first){u=x; first=false;}
	      else u=GROUP::space(u,x);
	    });
	  return u;
	});

    cnine::cachedf<CGprodBasisObj> reverse_standard_form=
      cnine::cachedf<CGprodBasisObj>([&](){
	  CGprodBasisObj* u; 
	  bool first=true;
	  for_each_leaf_reverse([&](CGprodBasisObj* x){
	      if(first){u=x; first=false;}
	      else u=GROUP::space(x,u);
	    });
	  return u;
	});


  public: // ---- Transformations to other bases -------------------------------------------------------------


    cnine::cachedf<EndMap<GROUP,double> > identity_map=
      cnine::cachedf<EndMap<GROUP,double> >([&](){
	  return new EndMap<GROUP,double>(tau,cnine::fill_identity());
	});

    cnine::cachedf<EndMap<GROUP,double> > standardizing_map=
      cnine::cachedf<EndMap<GROUP,double> >([&](){
	  if(is_leaf()) return  new EndMap<GROUP,double>(identity_map());
	  auto T=tprod(left->standardizing_map(),right->right_standardizing_map());
	  auto u=GROUP::space(&left->standard_form(),&right->reverse_standard_form());
	  
	  while(!u->right->is_leaf()){
	    T=u->left_shift_map()*T;
	    u=&u->shift_left();
	  }
	  return new EndMap<GROUP,double>(std::move(T));
	});

    cnine::cachedf<EndMap<GROUP,double> > right_standardizing_map=
      cnine::cachedf<EndMap<GROUP,double> >([&](){
	  if(is_leaf()) return new EndMap<GROUP,double>(tau,cnine::fill_identity());
	  auto T=tprod(left->standardizing_map(),right->right_standardizing_map());
	  auto u=GROUP::space(&left->standard_form(),&right->reverse_standard_form());
	  while(!u->left->is_leaf()){
	    T=cnine::transp(u->shift_right().left_shift_map())*T;
	    u=&u->shift_right();
	  }
	  return new EndMap<GROUP,double>(std::move(T));
	});

    cnine::cachedf<EndMap<GROUP,double> > left_shift_map=
      cnine::cachedf<EndMap<GROUP,double> >([&](){
	  auto R=new EndMap<GROUP,double>(tau,cnine::fill_zero());

	  GELIB_ASSRT(left && right);
	  GELIB_ASSRT(right->left && right->right);
	  CGprodBasisObj& x=*left;
	  CGprodBasisObj& y=*right->left;
	  CGprodBasisObj& z=*right->right;

	  for(auto& p1:x.tau){
	    auto l1=p1.first;
	    auto n1=p1.second;
	    for(auto& p2:y.tau){
	      auto l2=p2.first;
	      auto n2=p2.second;
	      for(auto& p3:z.tau){
		auto l3=p3.first;
		auto n3=p3.second;
		int m0=p1.second*p2.second*p3.second;
		//cout<<l1<<l2<<l3<<endl;
	    
		GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l12, const int m12){
		    GROUP::for_each_CGcomponent(l12,l3,[&](const _IrrepIx& l, const int mL){
			GROUP::for_each_CGcomponent(l2,l3,[&](const _IrrepIx& l23, const int m23){
			    GROUP::for_each_CGcomponent(l1,l23,[&](const _IrrepIx& ld, const int mR){
				if(l==ld){
				  //cout<<" "<<l12<<" "<<l23<<" "<<l<<endl;
				  auto& M=R->maps[l];
				  int loffs=shift_left().left_triple_offset(l1,l2,l3,l12,l);
				  int roffs=right_triple_offset(l1,l2,l3,l23,l);
				  int lwidth=m0*m12*mL;
				  int rwidth=m0*m23*mR;
				  GELIB_ASSRT(lwidth==rwidth);
				  //cout<<"("<<lwidth<<","<<rwidth<<")("<<loffs<<","<<roffs<<")"<<endl;
				  double v=GROUP::coupling(l1,l2,l3,l,l12,l23);
				  //cout<<l1<<","<<l2<<","<<l3<<","<<l<<":"<<l12<<","<<l23<<" "<<GROUP::coupling(l1,l2,l3,l,l12,l23)<<endl;
				  //auto T=cnine::Tensor<double>::constant({lwidth,rwidth},GROUP::coupling(l1,l2,l3,l,l12,l23));
				  //R->maps[l].block({lwidth,rwidth},{loffs,roffs})=T;
				  //for(int a=0; a<n1; a++)
				  //for(int b=0; b<n2; b++)
				  //  for(int c=0; c<n3; c++)
				  //M.set()
				  for(int i=0; i<lwidth; i++)
				    M.set(loffs+i,roffs+i,v);
				  //cout<<"."<<endl;
				}
			      });
			  });
		      });
		  });
	      }
	    }
	  }
	  return R;
	});

    cnine::cachedf<EndMap<GROUP,double> > right_shift_map=
      cnine::cachedf<EndMap<GROUP,double> >([&](){
	  return new EndMap<GROUP,double>(cnine::transp(shift_right().left_shift_map()));
	});

    cnine::cachedF<EndMap<GROUP,double> > swap_map=
      cnine::cachedF<EndMap<GROUP,double> >([&](){
	  if(is_leaf()) return identity_map();
 
	  EndMap<GROUP,double> R(tau,cnine::fill_zero());
	  auto& offsets=offset_map();
	  auto& swapped_offsets=swap().offset_map();

	  for(auto& p:left->tau)
	    for(auto& q:right->tau){
	      int l1=p.first;
	      int n1=p.second;
	      int l2=q.first;
	      int n2=q.second;

	      GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l, const int n){
		  auto& T=R.maps[l];
		  double c=GROUP::CG_sign_rule(l1,l2,l,0); // for now assume last index can be 0
		  int offs=offsets(l1,l2,l);
		  int swapped_offs=swapped_offs=swapped_offsets(l2,l1,l);
		  for(int i=0; i<n1; i++)
		    for(int j=0; j<n2; j++)
		      T.set(swapped_offs+i+j*n1,offs+i*n2+j,c);
		});
	    }

	  return R;
	});

    cnine::cachedF<EndMap<GROUP,double> > transpose_last_map=
      cnine::cachedF<EndMap<GROUP,double> >([&](){
	  if(is_leaf()) return identity_map(); 
	  if(is_stem()) return swap_map(); 
	  if(right->is_stem()) return tprod(left->identity_map(),right->swap_map());
	  if(right->is_leaf() && left->right->is_leaf()) 
	    return shift_right().transpose_last_map().conjugate(right_shift_map());
	  return standard_form().transpose_last_map().conjugate(standardizing_map());
	});
    //return cnine::transp(right_shift_map()) * shift_right().transpose_last_map() * right_shift_map();
    //return cnine::transp(standardizing_map()) * standard_form().transpose_last_map() * standardizing_map();

    cnine::cachedF<EndMap<GROUP,double> > lastJM=
      cnine::cachedF<EndMap<GROUP,double> >([&](){
	  if(is_leaf()) return identity_map(); 
	  if(is_stem()) return swap_map(); 
	  if(right->is_leaf())
	    return tprod(left->lastJM(),right->identity_map()).conjugate(transpose_last_map())+
	      transpose_last_map();
	  return standard_form().lastJM().conjugate(standardizing_map());
	});
    //return cnine::transp(standardizing_map()) * standard_form().last_JM() * standardizing_map();

	
  public: // ---- Index maps ---------------------------------------------------------------------------------


    int offset(const _IrrepIx l1, const _IrrepIx l2, const _IrrepIx l){
      return offset_map()(l1,l2,l);
    }

    int left_triple_offset(const _IrrepIx l1, const _IrrepIx l2, const _IrrepIx l3, const _IrrepIx l12, const _IrrepIx l){
      return left_triple_offset_map()(l1,l2,l3,l12,l);
    }

    int right_triple_offset(const _IrrepIx l1, const _IrrepIx l2, const _IrrepIx l3, const _IrrepIx l23, const _IrrepIx l){
      return right_triple_offset_map()(l1,l2,l3,l23,l);
    }


    typedef cnine::triple_map<_IrrepIx,_IrrepIx,_IrrepIx,int> OffsetMap;
    typedef cnine::quintuple_map<_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,int> TripleIndexMap;


    cnine::cachedf<OffsetMap> offset_map=
      cnine::cachedf<OffsetMap>([&](){
	  auto R=new OffsetMap();
	  Gtype<GROUP> offs;
	  for(auto& p:left->tau)
	    for(auto& q:right->tau)
	      GROUP::for_each_CGcomponent(p.first,q.first,[&](const _IrrepIx& l, const int n){
		  (*R)(p.first,q.first,l)=offs[l];
		  offs[l]+=n*p.second*q.second;
		});
	  return R;
	});

    cnine::cachedf<TripleIndexMap> left_triple_offset_map=
      cnine::cachedf<TripleIndexMap>([&](){
	  auto R=new TripleIndexMap();
	  CGprodBasisObj& x=*left->left;
	  CGprodBasisObj& y=*left->right;
	  CGprodBasisObj& z=*right;

	  Gtype<GROUP> offs;
	  for(auto& p1:x.tau){
	    auto l1=p1.first;
	    for(auto& p2:y.tau){
	      auto l2=p2.first;
	      GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l12, const int m12){
		  for(auto& p3:z.tau){
		    auto l3=p3.first;
		    GROUP::for_each_CGcomponent(l12,l3,[&](const _IrrepIx& l, const int m){
			(*R)(l1,l2,l3,l12,l)=offs[l];
			offs[l]+=m12*m*p1.second*p2.second*p3.second;
		      });
		  }
		});
	    }
	  }
	  return R;
	});
    
    cnine::cachedf<TripleIndexMap> right_triple_offset_map=
      cnine::cachedf<TripleIndexMap>([&](){
	  auto R=new TripleIndexMap();
	  GELIB_ASSRT(left && right->left && right->right);
	  CGprodBasisObj& x=*left;
	  CGprodBasisObj& y=*right->left;
	  CGprodBasisObj& z=*right->right;

	  Gtype<GROUP> offs;
	  for(auto& p1:x.tau){
	    auto l1=p1.first;
	    for(auto& p2:y.tau){
	      auto l2=p2.first;
	      for(auto& p3:z.tau){
		auto l3=p3.first;
		GROUP::for_each_CGcomponent(l2,l3,[&](const _IrrepIx& l23, const int m23){
		    GROUP::for_each_CGcomponent(l1,l23,[&](const _IrrepIx& l, const int m){
			(*R)(l1,l2,l3,l23,l)=offs[l];
			offs[l]+=m23*m*p1.second*p2.second*p3.second;
		      });
		  });
	      }
	    }
	  }
	  return R;
	});


  private: // ---- Internal ----------------------------------------------------------------------------------


    void add_counts(unordered_map<_IrrepIx,int>& counts) const{
      if(is_leaf()){
	counts[irrep]++;
	return;
      }
      left->add_counts(counts);
      right->add_counts(counts);
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
      //for(auto& p:isotypics)
      //oss<<p.second.str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CGprodBasisObj& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 


    /*
    for_each_Ltriple(std::function<void(const _IrrepIx&, const _IrrepIx&, const _IrrepIx&, const _IrrepIx&, const _IrrepIx&, const int, const int)>){
      CGprodBasisObj& x=*left->left;
      CGprodBasisObj& y=*left->right;
      CGprodBasisObj& z=*right;

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
    //CGprodBasisObj* Lmove() const{
    //return GROUP::FmoveL(this);
    //}

    //CGprodBasisObj* Rmove() const{
    //return GROUP::FmoveR(this);
    //}

    //const EndMap<GROUP,double>& LmoveMap(){
    //if(!_Lmove_map) make_Lmove_map();
    //return *_Lmove_map;
    //}
    /*
    pair<CGprodBasisObj*,const EndMap<GROUP,double>&> transform_left(){
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      GELIB_ASSRT(right->left!=nullptr && right->right!=nullptr);
      cout<<string("  ",indnt)<<"Left shifting "<<repr()<<endl;
      auto t=GROUP::space(left,right->left);
      return make_pair(GROUP::space(t,right->right),left_shift_map());
    }

    CGprodBasisObj* transform_right(){
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      GELIB_ASSRT(left->left!=nullptr && left->right!=nullptr);
      cout<<string("  ",indnt)<<"Right shifting "<<repr()<<endl;
      auto t=GROUP::space(left->right,right);
      return GROUP::space(left->left,t);
    }
    `*/

    /*
    CGprodBasisObj* cleave(){
      GELIB_ASSRT(left!=nullptr && right!=nullptr);
      //cout<<"Cleaving "<<repr()<<endl;
      CGprodBasisObj* u=left->standardize();
      CGprodBasisObj* v=right->reverse_standardize();
      return GROUP::space(u,v);
    }
    */
    /*
    const EndMap<GROUP,double>& standardizer(){
      if(!_standardizer) make_standardizer();
      return *_standardizer;
    }

    const EndMap<GROUP,double>& reverse_standardizer(){
      if(!_rstandardizer) make_rstandardizer();
      return *_rstandardizer;
    }
    */
    /*
    pair<CGprodBasisObj*, EndMap<GROUP,double> > standardize(){
      if(is_standard()) 
	return make_pair(this,EndMap<GROUP,double>(tau,cnine::fill_identity()));

      cout<<string("  ",indnt)<<"Standardizing "<<repr()<<endl; indnt++;

      auto x=left->standardize();
      auto y=right->reverse_standardize();
      auto u=GROUP::space(x.first,y.first);
      auto T=tprod(x.second,y.second);

      while(!u->right->is_leaf()){
	T=u->left_shift_map()*T;
	u=&u->shift_left();
      }

      indnt--; cout<<string("  ",indnt)<<"Standardized: "<<u->repr()<<endl; 
      return make_pair(u,T);
    }


    pair<CGprodBasisObj*, EndMap<GROUP,double> > reverse_standardize(){
      if(is_standard()) 
	return make_pair(this,EndMap<GROUP,double>(tau,cnine::fill_identity()));

      cout<<string("  ",indnt)<<"RStandardizing "<<repr()<<endl; indnt++;

      auto x=left->standardize();
      auto y=right->reverse_standardize();
      auto u=GROUP::space(x.first,y.first);
      auto T=tprod(x.second,y.second);

      while(!u->left->is_leaf()){
	T=cnine::transp(u->shift_right().left_shift_map())*T;
	u=&u->shift_right();
      }

      indnt--; cout<<string("  ",indnt)<<"RStandardized: "<<u->repr()<<endl;
      return make_pair(u,T);
    }
    */
    /*
    void make_standardizer(){
      if(_standardizer) return;
      if(is_standard()){
	_standardizer=new EndMap<GROUP,double>(tau,cnine::fill_identity());
	return;
      }
    }

    void make_rstandardizer(){
    }
    */
    //cnine::quintuple_map<_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,int>* Ltriple_index_map=nullptr;
    //cnine::quintuple_map<_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,_IrrepIx,int>* Rtriple_index_map=nullptr;
    /*
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
    */

    //GprodIndex<GROUP> index(const _IrrepIx& l, const int n){
    //GELIB_ASSRT(n<tau[l]);
    //auto 
    //}

    //int index(const GprodIndex& x){
    //GELIB_ASSRT(x.size()==nnodes);
    //}


	  /*
	  for_each_block([&](const _IrrepIx& l1, const _IrrepIx& l2, const _IrrepIx& l, int offs, int n){
	      auto& T=R.maps[l];
	      int swapped_offs=swapped_offs=swapped_offsets(l2,l1,l);
	      double c=GROUP::CG_sign_rule(l1,l2,l,0); // for now assume last index can be 0
	      for(int i=0; i<tau[l1]; i++)
		for(int j=0; i<tau[l2]; i++)
		T.set(swapped_offs+i,offs+i,c);
	      }
	    });
	  */
    /*
    void for_each_subisotypic_pair(std::function<void(const _Isotypic& I1, const _Isotypic& I2, const _Isotypic& I, int offs, int n)> lambda){
      GELIB_ASSRT(!is_leaf());
      CGprodBasisObj& x=*left;
      CGprodBasisObj& y=*right;

      for(auto& p1:x.tau){
	auto l1=p1.first;
	for(auto& p2:y.tau){
	  auto l2=p2.first;
	  GROUP::for_each_CGcomponent(l1,l2,[&](const _IrrepIx& l, const int m){
	      lambda(*x.isotypics[l1],*y.isotypics[l2],*isotypics[l],offset_map()(l1,l2,l),m); // *p1.second*p2.second);
	    });			
	}
      }
    }
    */
