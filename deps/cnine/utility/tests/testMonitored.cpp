#include "Cnine_base.cpp"
#include "monitored.hpp"
#include "CnineSession.hpp"


using namespace cnine;


class Widget;

obj_monitor<int> int_obj_monitor;


class Widget{
public:

  int id;

  monitored<int> member=monitored<int>(int_obj_monitor,[&](){
      //int* a=new int(2*id); 
      return make_shared<int>(2*id);});

  Widget(const int x):
    id(x){};

};


int main(int argc, char** argv){

  cnine_session session;

  Widget w1(1);
  Widget w2(2);
  Widget* w3=new Widget(3);
  cout<<int_obj_monitor<<endl;

  int b=w1.member;
  cout<<b<<endl;
  cout<<int_obj_monitor<<endl;

  int c=w2.member;
  cout<<c<<endl;
  cout<<int_obj_monitor<<endl;

  cout<<w3->member<<endl;
  cout<<int_obj_monitor<<endl;

  delete w3;
  cout<<int_obj_monitor<<endl;
}

