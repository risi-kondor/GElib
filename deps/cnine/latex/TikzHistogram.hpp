#ifndef _CnineTikzHistogram
#define _CnineTikzHistogram

#include "TikzStream.hpp"
#include "TikzTreeNode.hpp"


namespace cnine{


  class TikzHistogram{
  public:

    vector<double> vals;

    TikzHistogram(const vector<double>& _vals):
      vals(_vals){}


    string latex() const{
      TikzStream ots;
      ots<<"\\begin{tikzpicture}";
      ots<<"\\begin{axis}[ ymin=0, area style, xlabel=x, ylabel=y]\n";
      ots<<"\\addplot+[ybar interval,mark=no] plot coordinates {\n";
      for(int i=0; i<vals.size(); i++)
	ots<<"("<<i<<","<<vals[i]<<")";
      ots<<"};\n";
      ots.write("\\end{axis}");
      ots.write("\\end{tikzpicture}");
      return ots.str();
    }

  };

}


#endif 
