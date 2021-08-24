
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _YoungTableau
#define _YoungTableau

#include "IntegerPartition.hpp"

namespace GElib{


  class YoungTableau{
  public:

    vector<vector<int> > rows;


  public:

    YoungTableau(){};

    YoungTableau(const int n, const cnine::fill_identity& dummy){
      vector<int> row; 
      for(int i=0; i<n; i++) row.push_back(i+1);
      rows.push_back(row);
    }

    YoungTableau(const IntegerPartition& lambda){
      int t=1; 
      for(int r=0; r<lambda.height(); r++){
	vector<int> row(lambda[r]);
	for(int i=0; i<lambda[r]; i++) row[i]=t++;
	rows.push_back(row);
      }
    }

    YoungTableau(const initializer_list< initializer_list<int> > list){
      for(auto p:list){vector<int> v; for(auto pp:p) v.push_back(pp); rows.push_back(v);}
    }	


  public: // named constructors

  

  public: // Access
  
    int k() const {return rows.size();}

    vector<int> at(const int r) const{
      return rows.at(r);
    }
  
    int at(const int r, const int c) const{
      return rows.at(r).at(c);
    }

    IntegerPartition shape() const{
      IntegerPartition r(k(),cnine::fill_zero());
      for(int i=0; i<r.k; i++) r.add(i,rows[i].size());
      return r;
    }

    YoungTableau& add(const int r, const int j){
      if(r<k()){
	rows.at(r).push_back(j);
	return *this;
      }
      vector<int> row; 
      row.push_back(j); 
      rows.push_back(row); 
      return *this; 
    }

    YoungTableau& remove(const int r){
      if(at(r).size()>1){
	rows.at(r).pop_back(); 
	return *this;
      }
      rows.pop_back(); return *this;
    }

    bool operator==(const YoungTableau& x) {return rows == x.rows;}


  public: // I/O

    string str(const string indent="") const{
      ostringstream result;
      for(int i=0; i<rows.size(); i++){
	result<<indent;
	for(int j=0; j<rows[i].size(); j++) result<<rows[i][j]<<" ";
	result<<endl;
      }
      return result.str();
    }

    friend ostream& operator<<(ostream& stream, const YoungTableau& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif

