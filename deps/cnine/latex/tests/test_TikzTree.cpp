#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "TikzTree.hpp"
#include "LatexDoc.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  TikzTree ttree;

  auto root=ttree.add_root("");
  auto n1=root.add_child("1","\\texttt{for(i1=0; i1<n1; i1++)}");
  auto n2=root.add_child("2");
  auto n3=n1.add_child("3");

  auto st=ttree.latex();
  cout<<st<<endl;

  LatexDoc doc(ttree.latex());
  doc.compile("temp");
  system("open temp.pdf");

}
