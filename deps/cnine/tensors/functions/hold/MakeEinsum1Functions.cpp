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


#include "Ltensor.hpp"
#include "LtensorEinsum1.hpp"

int main(int argc, char** argv){

  ofstream ofs("Einsum1_loops.cpp");

  ofs<<"#include \"EinsumParams.hpp\""<<endl<<endl;

  ofs<<"namespace cnine{"<<endl<<endl;
  string indent="  ";

  for(int d=0; d<4; d++){

    for(int s=0; s<min(4,7-d); s++){

      for(int b=0; b<min(4,7-d); b++){
      
	ofs<<indent<<"template<typename TYPE>"<<endl;
	ofs<<indent<<"void Einsum1_D"<<to_string(d)<<"_S"<<to_string(s)<<"_B"<<to_string(b);
	ofs<<"(TYPE* r, const TYPE* x, const EsumParams& params){"<<endl;
	ofs<<endl;

	for(int i=0; i<d; i++)
	  ofs<<indent<<"  "<<"int I"<<to_string(i)<<"=params.ddims["<<to_string(i)<<"];"<<endl;
	for(int i=0; i<d; i++)
	  ofs<<indent<<"  "<<"int xstride_d"<<to_string(i)<<"=params.xstride_d["<<to_string(i)<<"];"<<endl;
	for(int i=0; i<d; i++)
	  ofs<<indent<<"  "<<"int rstride_d"<<to_string(i)<<"=params.rstride_d["<<to_string(i)<<"];"<<endl;

	for(int i=0; i<s; i++)
	  ofs<<indent<<"  "<<"int S"<<to_string(i)<<"=params.sdims["<<to_string(i)<<"];"<<endl;
	for(int i=0; i<s; i++)
	  ofs<<indent<<"  "<<"int xstride_s"<<to_string(i)<<"=params.xstride_s["<<to_string(i)<<"];"<<endl;

	for(int i=0; i<b; i++)
	  ofs<<indent<<"  "<<"int B"<<to_string(i)<<"=params.sdims["<<to_string(i)<<"];"<<endl;
	for(int i=0; i<b; i++)
	  ofs<<indent<<"  "<<"int rstride_b"<<to_string(i)<<"=params.rstride_b["<<to_string(i)<<"];"<<endl;
	ofs<<endl;

	for(int i=0; i<d; i++){
	  ofs<<indent<<"  "<<string(2*i,' ')<<"for(int i"<<to_string(i)<<"=0; i"<<to_string(i)<<"<I"<<to_string(i);
	  ofs<<"; i"<<to_string(i)<<"++){"<<endl;
	}
	ofs<<endl;


	// ---- Summation -------------------------------------------


	ofs<<indent<<"  "<<string(2*d,' ')<<"TYPE t=0;"<<endl;
	ofs<<indent<<"  "<<string(2*d,' ')<<"const TYPE* xslice=x+";
	for(int i=0; i<d; i++) 
	  ofs<<"i"<<to_string(i)<<"*xstride_d"<<to_string(i)<<"+";
	ofs<<"0;"<<endl;

	for(int i=0; i<s; i++){
	  ofs<<indent<<"  "<<string(2*i+2*d,' ')<<"for(int s"<<to_string(i)<<"=0; s"<<to_string(i)<<"<S"<<to_string(i);
	  ofs<<"; s"<<to_string(i)<<"++){"<<endl;
	}

	ofs<<indent<<"  "<<string(2*d+2*s,' ')<<"t+=xslice[";
	for(int i=0; i<s; i++) 
	  ofs<<"s"<<to_string(i)<<"*xstride_s"<<to_string(i)<<"+";
	ofs<<"0];"<<endl;

	for(int i=s-1; i>=0; i--)
	  ofs<<indent<<string(2*i+2*d+2,' ')<<"}"<<endl;
	ofs<<endl;

	
	// ---- Broadcasting --------------------------------------


	ofs<<indent<<"  "<<string(2*d,' ')<<"TYPE* rslice=r+";
	for(int i=0; i<d; i++) 
	  ofs<<"i"<<to_string(i)<<"*rstride_d"<<to_string(i)<<"+";
	ofs<<"0;"<<endl;

	for(int i=0; i<b; i++){
	  ofs<<indent<<"  "<<string(2*i+2*d,' ')<<"for(int b"<<to_string(i)<<"=0; b"<<to_string(i)<<"<B"<<to_string(i);
	  ofs<<"; b"<<to_string(i)<<"++){"<<endl;
	}

	ofs<<indent<<"  "<<string(2*d+2*b,' ')<<"rslice[";
	for(int i=0; i<b; i++) 
	  ofs<<"b"<<to_string(i)<<"*rstride_b"<<to_string(i)<<"+";
	ofs<<"0]+=t;"<<endl;

	for(int i=b-1; i>=0; i--)
	  ofs<<indent<<string(2*i+2*d+2,' ')<<"}"<<endl;
	//ofs<<endl;

	
	// --------------------------------------------------------


	for(int i=d-1; i>=0; i--)
	  ofs<<indent<<string(2*i+2,' ')<<"}"<<endl;
	
	ofs<<endl<<indent<<"}"<<endl<<endl<<endl;
      }

    }

  }
  ofs<<endl<<endl;

  ofs<<indent<<"void LtensorEinsum1loops(int D, int S, int B, float* r, const float* x, const EsumParams& params){"<<endl;
  ofs<<indent<<"  "<<"switch(D){"<<endl;
  for(int d=0; d<4; d++){
    ofs<<indent<<"  "<<"case "<<to_string(d)<<":"<<endl;
    ofs<<indent<<"    "<<"switch(S){"<<endl;
    for(int s=0; s<min(4,7-d); s++){
      ofs<<indent<<"    "<<"case "<<to_string(s)<<":"<<endl;
      ofs<<indent<<"      "<<"switch(B){"<<endl;
      for(int b=0; b<min(4,7-d); b++){
      ofs<<indent<<"      "<<"case "<<to_string(b)<<":"<<endl;
      ofs<<indent<<"        "<<"Einsum1_D"<<to_string(d)<<"_S"<<to_string(s)<<"_B"<<to_string(b)<<"(r,x,params);"<<endl;
      ofs<<indent<<"        break;"<<endl;
      }
      ofs<<indent<<"      "<<"}"<<endl;
      ofs<<indent<<"      break;"<<endl;
    }
    ofs<<indent<<"    "<<"}"<<endl;
    ofs<<indent<<"    break;"<<endl;
  }
  ofs<<indent<<"  "<<"}"<<endl;
  ofs<<indent<<"}"<<endl<<endl<<endl;

  ofs<<"}"<<endl;

  ofs.close();

}
