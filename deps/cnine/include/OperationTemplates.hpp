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


#ifndef _CnineOperationTemplates
#define _CnineOperationTemplates


namespace cnine{

  template<typename TENSORTYPE, typename = typename std::enable_if<std::is_base_of<CtensorA, TENSORTYPE>::value, TENSORTYPE>::type>
  TENSORTYPE operator*(complex<float> c, const TENSORTYPE& x){
    return x*c;
  }

  template<typename TENSORTYPE, typename = typename std::enable_if<std::is_base_of<CtensorA, TENSORTYPE>::value, TENSORTYPE>::type>
  TENSORTYPE operator*(CscalarObj& c, const TENSORTYPE& x){
    return x*c;
  }

  //template<typename TENSORTYPE, typename = typename std::enable_if<std::is_base_of<CtensorA, TENSORTYPE>::value, TENSORTYPE>::type>
  //TENSORTYPE operator*(CscalarObj& c, const TENSORTYPE& x){
  //return x*c;
  //}

}

#endif 
