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


#ifndef _CnineLtensorEinsum
#define _CnineLtensorEinsum

#include "Ltensor.hpp"


namespace cnine{


  class LtensorEinsum{
  public:

    int nargs=0;
    vector<vector<vector<int> > > diagonals;
    vector<vector<int> > remaps;
    vector<vector<int> > summations;
    vector<int> rdims;

    
    LtensorEinsum(const string str){

      vector<string> arg_str; 

      auto dout=str.find("->");
      if(dout==string::npos){
	COUT("Error in LtensorEinsum: malformed einsum string");
	return;
      }
      arg_str.push_back(str.substr(dout+2,string::npos));
      auto rest=str.substr(0,dout);

      int last=0;
      auto p=rest.find_first_of(',');
      while(p!=string::npos){
	arg_str.push_back(rest.substr(last,p));
	last=p+1;
	p=rest.find_first_of(',',last);
      }
      arg_str.push_back(rest.substr(last,string::npos));
      nargs=arg_str.size()-1;

      for(int i=0; i<nargs; i++)
	cout<<"arg"<<i<<": "<<arg_str[i+1]<<endl;
      cout<<"out:  "<<arg_str[0]<<endl;

      for(int j=0; j<nargs+1; j++){
	vector<vector<int> > diags; 
	auto& s=arg_str[j];
	for(int i=0; i<s.size(); i++){
	  char c=s[i];
	  if(s.find(c,i+1)!=string::npos){
	    auto v=find_all(s,c);
	    s=except(s,vector<int>(v.begin()+1,v.end()));
	    diags.push_back(v);
	  }
	}
	diagonals.push_back(diags);
      }

      auto& r_str=arg_str[0];
      for(int j=0; j<nargs; j++){
	auto& s=arg_str[j+1];
	vector<int> remap;
	vector<int> sums;
	int retard=0
	for(int i=0; i<s.size(); i++){
	  auto f=r_str.find(s[i]);
	  if(f!=string::npos)
	    remap.push_back(f);
	  else
	    sums.push_back[i-(retard++)];
	}
	remaps.push_back(remap);
	summations.push_back(sums);
      }

      //rdims=vector<int>(diagona)
      
      if(true){
	cout<<"Diags:"<<endl;
	for(auto& p:diagonals){
	  for(int i=0; i<p.size(); i++){
	    cout<<p[i];
	    if(i>0 && i<p.size()-1) cout<<",";
	  }
	  cout<<endl;
	}
	cout<<"Summations:"<<endl;
	for(int i=0; i<summations.size(); i++)
	  cout<<summations[i]<<endl;
	cout<<"Remappings:"<<endl;
	for(int i=0; i<remaps.size(); i++)
	  cout<<remaps[i]<<endl;
      }
      
    }

      
  private:

    inline vector<int> find_all(const string& str, const char c) const{
      vector<int> r;
      for(int i=0; i<str.size(); i++)
	if(str[i]==c) r.push_back(i);
      return r;
    }

    string except(string& s, vector<int> v) const{
      string r;
      int tail=0;
      for(int i=0; i<v.size(); i++){
	r.append(s.begin()+tail,s.begin()+v[i]);
	tail=v[i]+1;
      }
      return r.append(s.begin()+tail,s.end());
    }


  public:

    template<typename TYPE>
    Ltensor<TYPE> operator()(const Ltensor<TYPE>& _x){
      CNINE_ASSRT(nargs==1);
      auto x=diag(_x,diagonals[1]);
      return x;
    }


      
    template<typename TYPE>
    void add_to(Ltensor<TYPE>& R, const Ltensor<TYPE>& _x){

     Ltensor<TYPE> x=_x;
     for(auto& p:diagonals[1])
       x.reset(x.diag(p));

     for(auto& p:summations[1])
       x.reset(x.sum(p));

     auto& bcast=summations[0];
     for(int i=bcast.size()-1; i>=0; i--)
       x.reset(x.broadcast_explicit(bcast[i],3));
     
     Ltensor<TYPE> r(R);
     for(auto& p:diagonals[0])
       r.reset(r.diag(p));

     r+=x;
    }
      



  private: // -----------------------------------------------------------------------------------------------


    template<typename TYPE>
    Ltensor<TYPE> reduce(const Ltensor<TYPE>& _x, vector<vector<int> > ix){
      Ltensor<TYPE> x(_x);
      for(int i=0; i<ix.size(); i++){
	auto shrink=shrink_map(ix[i],x.ndims());
	x.reset(x.reduce(ix[i]));
	for(int j=i+1; j<ix.size(); j++)
	  shrink.shrink(ix[j]);
      }
    }

    template<typename TYPE>
    void broadcast_to_diagonal(const Ltensor<TYPE>& R, const Ltensor<TYPE>& x, const vector<vector<int> > ix){
      if(ix.size()==1){
	R.broadcast_to_diagonal(x,ix[0]);
	return;
      }

      Ltensor sub(R.dims.remove(ix.back()),R.get_dev());
      vector<vector<int> > subix;
      for(int i=0; i<ix.size()-1; i++)
	subix.push_back(shrink_map(ix.back(),R.ndims())(ix[i]));
      broadcast_to_diagonal(sub,x,subix);

      R.broadcast_to_diagonal(subix,ix.back());
    }



  };


}

#endif 

/*
  class shrink_map{
  public:

    vector<int> map;

    shrink_map(const vector<int>& ix, const int n):
      map(n){
      int tail=0;
      for(int i=0; i<ix.size(); i++){
	CNINE_ASSRT(ix[i]<n);
	for(int j=tail; j<ix[i]; j++)
	  map[j]=j-i;
	map[ix[i]]=-1;
	tail=ix[i]+1;
      }
      for(int j=tail; j<n; j++)
	map[j]=j-ix.size();
    }

    vector<int> operator()(const vector<int>& x){
      int N=x.size();
      vector<int> R(x.size());
      for(int i=0; i<N; i++){
	CNINE_ASSRT(R[i]<map.size());
	R[i]=map[R[i]];
      }
      return R;
    }

    void shrink(vector<int>& x){
      for(auto& p:x){
	CNINE_ASSRT(p<map.size());
	p=map[p];
      }
    }

  };
*/
  /*
    template<typename TYPE>
    Ltensor<TYPE> operator()(const Ltensor<TYPE>& _x){
      Ltensor<TYPE> x(_x);

      for(int i=0; i<sindices.size(); i++){
	auto& p=sindices[i];
	x.reset(x.diag(p[0]).sum(p[0][0]));
	for(int a=i+1; a<sindices.size(); a++)
	  for(int b=1; b<p[0].size(); b++)
	    for(int c=0; c<sindices[a][0].size(); b++)
	     if(sindices[a][0][c]>p[0][b]) sindices[a][0][c]--;
      }

      return x;
    }
  */
    /*
    template<typename TYPE>
    Ltensor<TYPE> diag(const Ltensor<TYPE>& _x, const vector<vector<int> >& diags){
      Ltensor<TYPE> x=_x;
      for(auto& p:diags)
	x.reset(x.diag(p));
      return x;
    }
    */
