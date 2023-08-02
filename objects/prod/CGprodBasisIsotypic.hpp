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

#include "cachedf.hpp"
#include "SnIsotypicSpace.hpp"
#include "IntegerPartition.hpp"
#include "IntegerPartitions.hpp"
#include "SnIrrep.hpp"
#include "BlockDiagonalize.hpp"
#include "CoupleIsotypics.hpp"
//#include "ColumnSpace.hpp"
#include "ComplementSpace.hpp"
#include "SingularValueDecomposition.hpp"


namespace GElib{

  template<typename GROUP>
  class CGprodBasisObj;


  template<typename GROUP>
  class CGprodBasisIsotypic{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef CGprodBasisIsotypic<GROUP> _Isotypic;
    typedef cnine::Tensor<double> _Tensor;
    typedef Snob2::IntegerPartition IP;

    _IrrepIx ix;
    int n=0;
    CGprodBasisObj<GROUP>* owner;
    map<Snob2::IntegerPartition,SnIsotypicSpace<double> > Snisotypics;


    ~CGprodBasisIsotypic(){
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    CGprodBasisIsotypic(){}

    CGprodBasisIsotypic(CGprodBasisObj<GROUP>* _owner, const _IrrepIx& _ix, const int _n=0):
      owner(_owner),ix(_ix), n(_n){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    CGprodBasisIsotypic(const CGprodBasisIsotypic& x)=delete;


  public: // ---- Maps --------------------------------------------------------------------------------------


    cnine::Tensor<double>& standardizing_map(){
      return owner->standardizing_map().maps[ix];
    }


  public: // ---- Sn-action ---------------------------------------------------------------------------------


    cnine::cachedf<map<IP,SnIsotypicSpace<double>* > > SnIsotypics= // Leak bc of pointers!
      cnine::cachedf<map<IP,SnIsotypicSpace<double>* > >([&](){
	  auto R=new map<IP,SnIsotypicSpace<double>* >;

	  if(owner->is_leaf()){
	    (*R)[{1}]=new SnIsotypicSpace<double>({1},cnine::Tensor<double>::identity({n,n}).split0(1,n));
	    return R;
	  }

	  if(owner->is_stem()){
	    int parity=GROUP::CG_sign_rule(owner->left->irrep,owner->right->irrep,ix,0); //TODO 
	    if(parity==1) 
	      (*R)[{2}]=new SnIsotypicSpace<double>({2},cnine::Tensor<double>::identity({n,n}).split0(1,n));
	    else
	      (*R)[{1,1}]=new SnIsotypicSpace<double>({1,1},cnine::Tensor<double>::identity({n,n}).split0(1,n));
	    return R;
	  }

	  GELIB_ASSRT(owner->right->is_leaf());
	  // the implicit assumption is that the right branch only has an isotypic corresponding to {1}
	  // it is also assumed that the multiplicity of each irrep in the CG-decomposition is 1

	  cout<<endl<<"Computing Sn-basis for "<<owner->repr()<<":"<<ix<<" [n="<<n<<"]"<<endl;

	  map<IP,int> multiplicities;
	  map<IP,int> dimensions;
	  owner->for_each_subisotypic_pair(ix,[&](_Isotypic& x, _Isotypic& y, int offs, int n){
	      const auto& subs=x.SnIsotypics();
	      for(auto& p:subs){
		multiplicities[p.first]+=p.second->dmult()*y.n;
		dimensions[p.first]=p.second->drho(); // this is a hack
	      }
	    });
	  
	  map<IP,SnIsotypicSpace<double> > induced;
	  for(auto& p:multiplicities)
	    induced[p.first]=SnIsotypicSpace<double>(p.first,dimensions[p.first],p.second,n,cnine::fill_zero()); 

	  map<IP,int> offsets;
	  owner->for_each_subisotypic_pair(ix,[&](_Isotypic& x, _Isotypic& y, int offs, int n){
	      const auto& xsubs=x.SnIsotypics();
	      const auto& ysubs=y.SnIsotypics();
	      for(auto& p:xsubs){
		IP ip=p.first;
		cout<<"Processing "<<ip<<endl;
		auto& xsub=*p.second;
		auto& ysub=*const_cast<map<IP,SnIsotypicSpace<double>* >&>(ysubs)[{1}]; // eliminate this by custom hash
		//cout<<dimensions[ip]<<","<<xsub.dmult()*y.n<<","<<x.n<<"*"<<y.n<<endl;
		//cout<<tprod(xsub,ysub).dims<<endl;
		//cout<<offsets[ip]<<" "<<offs<<" "<<induced[p.first].dims<<endl;
		induced[p.first].block({dimensions[ip],xsub.dmult()*y.n,x.n*y.n},{0,offsets[ip],offs})=
		  tprod(xsub,ysub);
		offsets[ip]+=xsub.dmult()*y.n;
	      }
	    });
	  
	  for(auto& p:induced)
	    cout<<""<<p.first<<":"<<p.second.dims<<endl;
	  cout<<endl;
	  
	  auto partitions=Snob2::IntegerPartitions::RestrictionOrdered(induced.begin()->first.getn()+1);
	  for(auto lambda:partitions){
	    cout<<"Searching for "<<lambda<<endl;
	    
	    vector<IP> subs=lambda.parents();
	    if(any_of(subs.begin(),subs.end(),[&](const IP& ip){
		  return induced.find(ip)==induced.end();})) continue;

	    auto joint=cnine::cat<vector<IP>,IP,double>(0,subs,[&](const IP& x){
		return induced[x].matrix();});
	    auto projected=joint*(owner->transpose_last_map().maps[ix]*(joint.transp()));
	    
	    auto basis1=cnine::ComplementSpace<double>(projected-cnine::Tensor<double>::identity_like(projected))();
	    auto basis2=cnine::ComplementSpace<double>(projected+cnine::Tensor<double>::identity_like(projected))();
	    auto basis=cat(1,basis1,basis2);

	    int rdim=Snob2::SnIrrep(lambda).dim();
	    int m=basis.dims[1]/rdim;
	    if(m==0) continue;
	    (*R)[lambda]=new SnIsotypicSpace<double>(lambda,(basis.transp()*joint).split0(rdim,m));
	    cout<<"  Found "<<((double)basis.dims[1])/rdim<<" copies of rho"<<lambda<<endl;

	    joint=(cnine::Tensor<double>::identity({joint.dims[0],joint.dims[0]})-(basis*basis.transp()))*joint;
	    int offs=0;
	    for(auto p:subs){
	      auto M=induced[p].matrix();
	      M=joint.rows(offs,offs+M.dims[0]);
	      offs+=M.dims[0];
	    }

	  }

	  return R;
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

    friend ostream& operator<<(ostream& stream, const CGprodBasisIsotypic& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 


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

    /*
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
    */

	    /*
	    cnine::Tensor<double> M({projected.dims[0],projected.dims[0]},cnine::fill_identity());
	    int prev_dims;
	    do{
	      prev_dims=M.dims[1];
	      cout<<M.str("in:")<<endl;
	      M=cnine::ColumnSpace(projected*M)();
	      cout<<M.str("out:")<<endl;
	      cout<<M.dims[1]<<endl;
	    }while(M.dims[1]<prev_dims);
	    */
	    //int rhodim=0;
	    /*
	    int offs=0;
	    for(auto p:subs){
	      auto inducedM=induced[p].matrix();
	      cout<<1<<endl;
	      cout<<" "<<inducedM.dims<<endl;
	      cout<<" "<<remaining.rows(offs,offs+inducedM.dims[0]).dims<<endl;
	      inducedM=remaining.block({offs,offs+inducedM.dims[0]).transp()*inducedM;
	      cout<<2<<endl;
	      offs+=inducedM.dims[0];
	    }
	    */
    // deprecated 
    /*
    void make_Snisotypics(){
      if(Snisotypics.size()) return;

      if(!owner->is_standard()){
	for(auto& p: owner->standard_form().isotypics){
	  p.second->make_Snisotypics();
	  for(auto& q:p.second->Snisotypics)
	    Snisotypics[q.first]=q.second.transform(standardizing_map().transp());
	}
	return;
      }

      if(owner->is_leaf()){
	Snisotypics[{1}]=SnIsotypicSpace<double>({1},cnine::Tensor<double>::identity({n,n}).split0(1,n));
	return;
      }

      if(owner->is_stem()){
	int parity=GROUP::CG_sign_rule(owner->left->irrep,owner->right->irrep,ix,0); //TODO 
	if(parity==1) 
	  Snisotypics[{2}]=SnIsotypicSpace<double>({2},cnine::Tensor<double>::identity({n,n}).split0(1,n));
	else
	  Snisotypics[{1,1}]=SnIsotypicSpace<double>({1,1},cnine::Tensor<double>::identity({n,n}).split0(1,n));
	return;
      }

      cout<<"Making isotypics for "<<owner->repr()<<": l="<<ix<<endl;

      for(auto& p:owner->left->isotypics)
	p.second->make_Snisotypics();

      map<IP,int> counts;
      map<IP,int> drep; // this is a crutch
      owner->for_each_block([&](const _IrrepIx& l1, const _IrrepIx& l2, const _IrrepIx& l, int offs, int n){
	  if(l!=ix) return;
	  for(auto& p:owner->left->isotypics[l1]->Snisotypics){
	    counts[p.first]+=p.second.multiplicity();
	    drep[p.first]=p.second.drep();
	  }});

      map<IP,SnIsotypicSpace<double> > subisotypics;
      for(auto& p:counts)
	subisotypics[p.first]=SnIsotypicSpace<double>(p.first,drep[p.first],p.second,n,cnine::fill_zero());

      map<IP,int> roffs;
      owner->for_each_block([&](const _IrrepIx& l1, const _IrrepIx& l2, const _IrrepIx& l, int offs, int n){
	  if(l!=ix) return;
	  for(auto& p:owner->left->isotypics[l1]->Snisotypics){
	    cnine::Tensor<double>& A(p.second);
	    GELIB_ASSRT(n==A.dims[2]);
	    subisotypics[p.first].block(A.dims,{0,roffs[p.first],offs})=A;
	    roffs[p.first]+=p.second.multiplicity();
	  }
	});

      auto M=owner->transpose_last_map().maps[ix];
      cout<<M<<endl;
      cnine::BlockDiagonalize blocked(M);

      vector<_Tensor> spaces;
      int doff=0;
      for(auto d:blocked.sizes){
	spaces.push_back(blocked.V.block({n,d},{0,doff}).transp());
	doff+=d;
	cout<<spaces.back()<<endl;
      }

      CoupleIsotypics(subisotypics,spaces);
    }
    */
