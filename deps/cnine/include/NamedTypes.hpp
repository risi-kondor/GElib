/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineNamedTypes
#define _CnineNamedTypes

#include "NamedType.hpp"
#include "Gdims.hpp"


namespace cnine{

  using BatchArgument=NamedType<int, struct BatchArgumentTag>;
  using GridArgument=NamedType<Gdims, struct GridArgumentTag>;
  using DimsArgument=NamedType<Gdims, struct DimsArgumentTag>;
  using ChannelsArgument=NamedType<int, struct ChannelsArgumentTag>;
  using FillArgument=NamedType<int, struct FillArgumentTag>;
  using DeviceArgument=NamedType<int, struct DeviceArgumentTag>;
  using DtypeArgument=NamedType<dtype_enum, struct DtypeArgumentTag>;

  static const BatchArgument::argument batch;
  static const GridArgument::argument grid;
  static const DimsArgument::argument cdims;
  static const ChannelsArgument::argument channels;
  static const FillArgument::argument filltype;
  static const DeviceArgument::argument device;
  static const DtypeArgument::argument dtype;


}

#endif 
