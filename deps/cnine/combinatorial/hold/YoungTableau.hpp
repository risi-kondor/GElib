#ifndef _CnineYoungTableau
#define _CnineYoungTableau

#include "IntegerPartition.hpp"

namespace cnine{


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

    int getk() const {return rows.size();}

    int nrows() const {return rows.size();}

    IntegerPartition shape() const{
      IntegerPartition r(k(),cnine::fill_zero());
      for(int i=0; i<r.k; i++) r.add(i,rows[i].size());
      return r;
    }

    vector<int> at(const int r) const{
      return rows.at(r);
    }
  
    int at(const int r, const int c) const{
      return rows.at(r).at(c);
    }

    int operator()(const int r, const int c) const{
      return rows.at(r).at(c);
    }

    int get_value(const int r, const int c) const{
      return rows.at(r).at(c);
    }

    int set_value(const int r, const int c, const int x){
      return rows.at(r)[c]=x;
    }

    pair<int,int> index(const int m) const{
      for(int i=0; i<=k()-1; i++) {
	for(int j=0; j<rows.at(i).size(); j++)
	  if(at(i,j)==m) return pair<int,int>(i,j);
      }
      return pair<int,int>(0,0); // should never happen
    }

    bool operator==(const YoungTableau& x) {return rows == x.rows;}


  public:

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

    int apply_transp(const int i){
      return apply_transp(i,i+1);
    } 

    int apply_transp(const int i, const int j){
      pair<int,int> rc1=index(i);
      pair<int,int> rc2=index(j);
      rows.at(rc1.first).at(rc1.second)=j;
      rows.at(rc2.first).at(rc2.second)=i;
      return (rc2.second-rc2.first)-(rc1.second-rc1.first);
    }


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

