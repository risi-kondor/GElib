#include "Cnine_base.cpp"
#include "observable.hpp"
#include "CnineSession.hpp"


using namespace cnine;


class Target: public observable<Target>{
public:
  
  string name;
  
  Target(const string _name):
    observable<Target>(this),
    name(_name){}

};


class Follower{
public:

  string name;
  observer<Target> observer;

  Follower(const string _name, Target* target):
    name(_name), 
    observer(target,[&](Target* x){report(x);}){}
    //observer(target){}

  ~Follower(){
    cout<<name<<" deleted."<<endl;
  }

  void report(Target* target){
    cout<<"Target "<<target->name<<" reported as deleted ("<<name<<")."<<endl;
  }

};


int main(int argc, char** argv){

  cnine_session session;

  Target* t0=new Target("Target0");

  Follower* A=new Follower("FollowerA",t0);
  Follower* B=new Follower("FollowerB",t0);
  Follower* C=new Follower("FollowerC",t0);

  delete C;

  delete t0;


}

