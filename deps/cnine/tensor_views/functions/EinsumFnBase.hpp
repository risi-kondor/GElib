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


#ifndef _EinsumFnBase
#define _EinsumFnBase

#include "CtensorView.hpp"


namespace cnine{


  class EinsumFnBase{
  public:

    vector<pair<vector<int>,vector<int> > > sstrides;
    vector<triple<vector<int> > > dstrides;
    vector<vector<int> > bstrides;

    bool rconj;
    bool xconj;
    bool yconj;

    EinsumFnBase(const string str){
      auto d0=str.find(",");
      auto d1=str.find("->");
      if(d0==string::npos || d1==string::npos || d0>d1){
	COUT("Error in CtensorEinsumFn: malformed einsum string");
	return;
      }
      auto xstr=str.substr(0,d0);
      auto ystr=str.substr(d0+1,d1-d0-1);
      auto rstr=str.substr(d1+2,string::npos);
      //cout<<xstr<<endl;
      //cout<<ystr<<endl;
      //cout<<rstr<<endl;

      rconj=is_conj(rstr);
      xconj=is_conj(xstr);
      yconj=is_conj(ystr);
      if(rconj){
	xconj=!xconj;
	yconj=!yconj;
      }

      while(true){
	auto p=rstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=rstr[p];
	auto rindices=find_all(rstr,c);

	if(xstr.find(c)==string::npos && ystr.find(c)==string::npos){ // broadcast case
	  bstrides.push_back(rindices);
	}else{ // sirect case
	  dstrides.push_back(triple<vector<int> >(rindices,find_all(xstr,c),find_all(ystr,c)));
	}
      }

      while(true){
	auto p=xstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=xstr[p];
	sstrides.push_back(pair<vector<int>,vector<int> >(find_all(xstr,c),find_all(ystr,c)));
      }
      
      if(false){
	if(xconj) cout<<"*"; else cout<<"x";
	if(yconj) cout<<"*"; else cout<<"x";
	cout<<"->";
	if(rconj) cout<<"*"; else cout<<"x";
	cout<<endl;
	cout<<"Direct:"<<endl;
	for(auto& p:dstrides){
	  cout<<p.second<<","<<p.third<<"->"<<p.first<<endl;
	}
	cout<<"Sum:"<<endl;
	for(auto& p:sstrides){
	  cout<<p.first<<","<<p.second<<endl;
	}
	cout<<"Broadcast:"<<endl;
	for(auto& p:bstrides){
	  cout<<p<<endl;
	}
      }

    }


    bool is_conj(string& str){
      if(str.back()=='*'){
	str.replace(str.size()-1,1,1,'x');
	return true;
      }
      return false;
    }


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


  };

}

#endif
