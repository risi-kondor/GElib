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


  //template<typename TYPE>
  class LtensorEinsum{
  public:

    vector<vector<vector<int> > > sindices;
    vector<vector<vector<int> > > dstrides;
    vector<vector<int> > bstrides;


    LtensorEinsum(const string str){
      auto dout=str.find("->");
      if(dout==string::npos){
	COUT("Error in LtensorEinsum: malformed einsum string");
	return;
      }
      auto args_str=str.substr(0,dout);
      auto rstr=str.substr(dout+2,string::npos);

      //vector<int> darg; 
      vector<string> arg; 
      int last=0;
      auto p=args_str.find_first_of(',');
      while(p!=string::npos){
	arg.push_back(str.substr(last,p));
	last=p+1;
	p=args_str.find_first_of(',',last);
      }
      arg.push_back(args_str.substr(last,string::npos));

      for(int i=0; i<arg.size(); i++)
	cout<<"arg"<<i<<": "<<arg[i]<<endl;
      cout<<"out:  "<<rstr<<endl;

      while(true){
	auto p=rstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=rstr[p];
	auto rstrides=find_all(rstr,c);
	cout<<rstrides<<endl;

	if([&](){for(auto& p:arg) if (p.find(c)!=string::npos) return true; return false;}()){
	  vector<vector<int> > directs;
	  directs.push_back(rstrides);
	  for(auto& p:arg)
	    directs.push_back(find_all(p,c));
	  dstrides.push_back(directs);
	}else{
	  bstrides.push_back(rstrides);
	}	  
      }

      while(true){
	auto p=arg[0].find_first_not_of('x');
	if(p==string::npos) break;
	char c=arg[0][p];
	vector<vector<int> > v;
	v.push_back(find_all(arg[0],c));
	sindices.push_back(v);
      }
      
      if(true){
	cout<<"Direct:"<<endl;
	for(auto& p:dstrides){
	  for(int i=0; i<p.size(); i++){
	    cout<<p[i];
	    if(i==0) cout<<"<-";
	    if(i>0 && i<p.size()-1) cout<<",";
	  }
	  cout<<endl;
	}
	cout<<"Sum:"<<endl;
	for(auto& p:sindices){
	  for(int i=0; i<p.size(); i++){
	    cout<<p[i];
	    if(i>0 && i<p.size()-1) cout<<",";
	  }
	  cout<<endl;
	}
	cout<<"Broadcast:"<<endl;
	for(auto& p:bstrides){
	  cout<<p<<endl;
	}
      }
      
    }

      
  private:

    inline vector<int> find_all(string& str, const char c) const{
      vector<int> r;
      auto p=str.find_first_of(c);
      while(p!=string::npos){
	str.replace(p,1,1,'x');
	r.push_back(p);
	p=str.find_first_of(c);
      }
      return r;
    }

  public:

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

  };


}

#endif 
