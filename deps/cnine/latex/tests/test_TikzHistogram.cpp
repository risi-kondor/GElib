#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "TikzHistogram.hpp"
#include "LatexDoc.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  vector<double> data({13,40,29.6,7,30,10,2,2,40});

  TikzHistogram hist(data);

  LatexDoc doc(hist.latex());
  doc.compile("hist");
  system("open hist.pdf");

}
