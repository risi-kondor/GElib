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
#include "ComplementSpace.hpp"
#include "IntersectionSpace.hpp"
#include "SymmEigenspace.hpp"
#include "SingularValueDecomposition.hpp"
#include "SymmEigendecomposition.hpp"
#include "MakeCoherentSnIsotypic.hpp"


namespace GElib{

  template<typename GROUP>
  class CGprodBasisObj;


  template<typename GROUP>
  class CGprodBasisIsotypic{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef CGprodBasisIsotypic<GROUP> _Isotypic;
    typedef cnine::Tensor<double> Tensor;
    typedef Snob2::IntegerPartition IP;

    _IrrepIx ix;
    int n=0;
    CGprodBasisObj<GROUP>* owner;


    ~CGprodBasisIsotypic(){
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    CGprodBasisIsotypic(){}

    CGprodBasisIsotypic(CGprodBasisObj<GROUP>* _owner, const _IrrepIx& _ix, const int _n=0):
      owner(_owner),ix(_ix), n(_n){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    CGprodBasisIsotypic(const CGprodBasisIsotypic& x)=delete;


  public: // ---- Maps --------------------------------------------------------------------------------------


    Tensor& standardizing_map(){
      return owner->standardizing_map().maps[ix];
    }


  public: // ---- Sn-action ---------------------------------------------------------------------------------


    cnine::cachedf<map<IP,SnIsotypicSpace<double> > > Snisotypics= 
      cnine::cachedf<map<IP,SnIsotypicSpace<double> > >([&](){

	  auto R=new map<IP,SnIsotypicSpace<double> >;

	  if(owner->is_leaf()){
	    (*R)[{1}]=SnIsotypicSpace<double>({1},cnine::Identity<double>(n).split0(1,n));
	    return R;
	  }

	  if(owner->is_stem()){
	    int parity=GROUP::CG_sign_rule(owner->left->irrep,owner->right->irrep,ix,0); //TODO 
	    if(parity==1) 
	      (*R)[{2}]=SnIsotypicSpace<double>({2},cnine::Identity<double>(n).split0(1,n));
	    else
	      (*R)[{1,1}]=SnIsotypicSpace<double>({1,1},cnine::Identity<double>(n).split0(1,n));
	    return R;
	  }

	  GELIB_ASSRT(owner->right->is_leaf());
	  // the implicit assumption is that the right branch only has an isotypic corresponding to {1}
	  // it is also assumed that the multiplicity of each irrep in the CG-decomposition is 1

	  cout<<endl<<"Computing Sn-basis for "<<owner->repr()<<":"<<ix<<" [n="<<n<<"]"<<endl;
	  const auto JM=owner->lastJM()[ix];


	  // ---- Gather S_{n-1} representations

	  map<IP,int> submultiplicities;
	  owner->for_each_subisotypic_pair(ix,[&](_Isotypic& x, _Isotypic& y, int offs, int n){
	      for(auto& [lambda,iso]: x.Snisotypics()){
		//auto lambda=p.first;
		//auto iso=p.second;
		submultiplicities[lambda]+=iso.dmult()*y.n;
	      }
	    });
	  
	  map<IP,SnIsotypicSpace<double> > induced;
	  for(auto& [lambda,m]: submultiplicities){
	    //auto lambda=p.first;
	    //auto m=p.second;
	    induced.emplace(lambda,
	      SnIsotypicSpace<double>(lambda,Snob2::SnIrrep(lambda).dim(),m,n,cnine::fill_zero())); 
	  }

	  map<IP,int> offsets;
	  owner->for_each_subisotypic_pair(ix,[&](_Isotypic& x, _Isotypic& y, int offs, int n){
	      const auto& xsubs=x.Snisotypics();
	      const auto& ysubs=y.Snisotypics();
	      for(auto& [ip,xsub]: xsubs){
		//auto ip=p.first;
		//auto xsub=p.second;
		auto& ysub=const_cast<map<IP,SnIsotypicSpace<double> >& >(ysubs)[{1}]; // eliminate this by custom hash
		induced[ip].block({Snob2::SnIrrep(ip).dim(),xsub.dmult()*y.n,x.n*y.n},{0,offsets[ip],offs})=
		  tprod(xsub,ysub);
		offsets[ip]+=xsub.dmult()*y.n;
	      }
	    });


	  // ---- Compute multiplicities for new SnIsotypicSpaces

	  map<IP,int> multiplicities;
	  auto partitions=Snob2::IntegerPartitions(induced.begin()->first.getn()+1);
	  for(auto lambda:partitions){
	    //cout<<"Searching for "<<lambda<<endl;
	    
	    vector<IP> subs=lambda.parents();
	    if(any_of(subs.begin(),subs.end(),[&](const IP& ip){
		  return induced.find(ip)==induced.end();})) continue;

	    for(auto mu:subs){
	      auto E=cnine::SymmEigenspace<double>(JM,lambda.content_of_difference(mu))();
	      auto S=cnine::IntersectionSpace<double>(E.transp(),induced[mu].slice(0,0))();

	      if(multiplicities.find(lambda)==multiplicities.end())
		multiplicities.emplace(lambda,S.dims[0]);
	      else{
		if(S.dims[0]!=multiplicities[lambda]){
		  cout<<"Error: Inconsistent multiplicities in gathered basis."<<endl;
		  return R;
		}
	      }

	    }

	  }
	   
	  int i=0;
	  cout<<"  ";
	  for(auto& [lambda,m]: multiplicities)
	    if(m>0){
	      if(i++>0) cout<<"+";
	      cout<<m<<"*"<<lambda;
	    }
	  cout<<endl;


	  // ---- Compute new bases 


	  for(auto& p: multiplicities){
	    auto lambda=p.first;
	    auto M=p.second;
	    if(M==0) continue;
	    //cout<<"lambda="<<lambda<<":"<<endl;
	    
	    map<IP,SnIsotypicSpace<double> > sources;
	    lambda.for_each_sub([&](const IP& mu){
		sources[mu]=SnIsotypicSpace<double>(mu,Snob2::SnIrrep(mu).dim(),M,n,cnine::fill_zero());
		auto E=cnine::SymmEigenspace<double>(JM,lambda.content_of_difference(mu))();
		auto S=cnine::IntersectionSpace<double>(E.transp(),induced[mu].slice(0,0))();
		auto P=S*(induced[mu].slice(0,0).transp());
		for(int i=0; i<Snob2::SnIrrep(mu).dim(); i++)
		  sources[mu].slice(0,i)=P*induced[mu].slice(0,i);
	      });

	    (*R)[lambda]=MakeCoherentSnIsotypic(lambda,sources,owner->transpose_last_map().maps[ix])();
	  }

	  return R;

	});

   

    cnine::cachedf<map<IP,SnIsotypicSpace<double>* > > SnIsotypics= // Leak bc of pointers!
      cnine::cachedf<map<IP,SnIsotypicSpace<double>* > >([&](){
	  auto R=new map<IP,SnIsotypicSpace<double>* >;

	  if(owner->is_leaf()){
	    (*R)[{1}]=new SnIsotypicSpace<double>({1},cnine::Identity<double>(n).split0(1,n));
	    return R;
	  }

	  if(owner->is_stem()){
	    int parity=GROUP::CG_sign_rule(owner->left->irrep,owner->right->irrep,ix,0); //TODO 
	    if(parity==1) 
	      (*R)[{2}]=new SnIsotypicSpace<double>({2},cnine::Identity<double>(n).split0(1,n));
	    else
	      (*R)[{1,1}]=new SnIsotypicSpace<double>({1,1},cnine::Identity<double>(n).split0(1,n));
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
		//cout<<"Processing "<<ip<<endl;
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
	  
	  for(auto& p:induced){
	    cout<<""<<p.first<<":"<<p.second.dims<<endl;
	    //cout<<p.second.matrix().str("  ")<<endl;
	  }
	  cout<<endl;
	  
	  auto partitions=Snob2::IntegerPartitions::RestrictionOrdered(induced.begin()->first.getn()+1);
	  for(auto lambda:partitions){
	    //cout<<"Searching for "<<lambda<<endl;
	    
	    vector<IP> subs=lambda.parents();
	    if(any_of(subs.begin(),subs.end(),[&](const IP& ip){
		  return induced.find(ip)==induced.end();})) continue;
	    //cout<<1<<endl;

	    auto joint=cnine::cat<vector<IP>,IP,double>(0,subs,[&](const IP& x){
		return induced[x].matrix();});
	    auto projected=joint*(owner->transpose_last_map().maps[ix]*(joint.transp()));
	    //cout<<owner->transpose_last_map().maps[ix]<<endl;
	    //cout<<joint<<endl;
	    //cout<<projected<<endl;
	    //cout<<2<<endl;

	    auto basis1=cnine::ComplementSpace<double>(projected-Tensor::identity_like(projected))();
	    auto basis2=cnine::ComplementSpace<double>(projected+Tensor::identity_like(projected))();
	    auto basis=cat(1,basis1,basis2);
	    //cout<<joint.str("joint:")<<endl;
	    //cout<<basis<<endl;
	    //cout<<basis.transp()*basis<<endl;
	    //cout<<3<<endl;

	    int rdim=Snob2::SnIrrep(lambda).dim();
	    //cout<<4<<endl;
	    int m=basis.dims[1]/rdim;
	    if(m==0) continue;
	    Tensor TB=basis.transp();
	    Tensor A=TB*joint;
	    //cout<<TB.str("TB=")<<endl;
	    //cout<<A.str("A=")<<endl;
	    //Tensor A=basis.transp()*joint;
	    //Tensor C=A.split0(rdim,m);
	    cout<<"  Found "<<((double)basis.dims[1])/rdim<<" copies of rho"<<lambda<<endl;
	    (*R)[lambda]=new SnIsotypicSpace<double>(lambda,A.split0(rdim,m));
	    //cout<<(*R)[lambda]->matrix().str("found:")<<endl;
	    //(*R)[lambda]=new SnIsotypicSpace<double>(lambda,(basis.transp()*joint).split0(rdim,m));
	    //cout<<5<<endl;

	    if(lambda.getn()==4 && ix==2){cout<<"Skipping"<<endl; continue;}
	    joint=(Tensor::identity({joint.dims[0],joint.dims[0]})-(basis*basis.transp()))*joint;
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
    //map<Snob2::IntegerPartition,SnIsotypicSpace<double> > Snisotypics;
	  //cout<<owner->lastJM()[ix]<<endl;
	  //cnine::SingularValueDecomposition svd(owner->lastJM()[ix]);
	  //cout<<svd.S()<<endl;

	  //cout<<owner->transpose_last_map()[ix]<<endl;
	  //cout<<"TLM error: "<<owner->transpose_last_map()[ix].unitary_error()<<endl;
	  //cout<<owner->swap_map()[ix]<<endl;

	  //map<IP,int> dimensions;
	  //owner->for_each_subisotypic_pair(ix,[&](_Isotypic& x, _Isotypic& y, int offs, int n){
	  //  for(auto& p:x.Snisotypics())
	  //if(dimensions.find(p.first)==dimensions.end()) dimensions.emplace(p.first,p.second.drho());
	  //});
	  
	  //int tdims=0;
	  //for(auto& [lambda,m]: multiplicities){
	    //if(m>0) R->emplace(lambda,SnIsotypicSpace<double>(lambda,Snob2::SnIrrep(lambda).dim(),m,n,cnine::fill_zero())); 
	    //tdims+=Snob2::SnIrrep(lambda).dim();
	  //}

	    /*
	    vector<IP> subs=lambda.parents();
	    map<IP,int> offs;
	    int t=0;
	    for(auto p:subs){
	      offs.emplace(p,t);
	      t+=Snob2::SnIrrep(p).dim();
	    }
	      
	    auto mu0=subs[0];
	    auto E0=cnine::SymmEigenspace<double>(JM,lambda.content_of_difference(mu0))();
	    auto S0=cnine::IntersectionSpace<double>(E.transp(),induced[mu0].slice(0,0))();
	    //auto V=S0*(owner->transpose_last_map().maps[ix]);
	    GELIB_ASSRT(S0.dims[0]==M);


	    for(int m=0; m<M; m++){ // for each copy
	      Tensor<double> B({Snob2::SnIrrep(lambda).dim(),n},cnine::fill_zero);

	      auto coeffs=(induced[m0].slice(0,0))*(S0.row(m).transp());
	      for(int d=0; d<Snob2::SnIrrep(mu).dim(); d++)
		B.row(d)=coeffs*induced[m0].slice(0,d);
	      (*R)[lambda].slice(1,m)=B.rows(0,Snob2::SnIrrep(mu).dim());

	      std::set<IP> remaining;
	      for(int i=1; i<subs.size(); i++)
		remaining.insert(subs[i]);

	      while(remaining.size()>0){
		auto D=B;
	      }

	    */
	    //R->emplace(lambda,SnIsotypicSpace<double>(lambda,Snob2::SnIrrep(lambda).dim(),M,n,cnine::fill_zero())); 
    /*
    void add_mu_subspace_to_isotypic(const IP& lambda, const IP& mu, int m, TensorView<double>& coeffs){
      GELIB_ASSRT(coeffs.dims[0]==M);

      for(int d=0; d<Snob2::SnIrrep(mu).dim(); d++)
	(*R)[lambda].slice(1,m).row(offs[mu]+d)=coeffs*(sources[mu].slice(0,d)); 
      remaining.erase(mu);

      if(remaining.size()>0){

	auto S=RowSpace( (*R)[lambda].slice(1,m) * owner->transpose_last_map().maps[ix] )(); 
	map<IP,Tensor<double> > projection_vectors;

	for(auto nu:remaining){

	  Tensor<double> v({M},cnine::fill_zero());
	  for(int i=0; i<d; i++){
	    auto Pi=Project(S,sources[nu].slice(0,d))();
	    v=v+Pi.slice(0,0);
	  }
	  if(v.norm
	  projection_vectors.emplace(nu,v);

	}

	for(
	

	  if(P.norm()>10e-5){
	    P.normalize();
	    complete_isotypic(lambda,nu, 
	  }
	  

	}
      }
    }

    */
