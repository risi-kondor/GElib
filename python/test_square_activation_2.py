import torch
import time

import sys
sys.path.insert(1, '../models/')
import gelib_torchC as gelib

torch.manual_seed(123)

# Define the type 
batch = 20
input_maxl = 11
hidden_maxl = 3

# Input signal
input_signal = gelib.SO3vec.randn(batch, [1 for l in range(input_maxl + 1)])

print("\nInput signal ------------------------------------------")
for l in range(len(input_signal.parts)):
    print(input_signal.parts[l].size())

# print(input_signal)

# Initialize a filter 
f = gelib.SO3vec.Frandn(batch, hidden_maxl)

print("\nFilter ------------------------------------------------")
for l in range(len(f.parts)):
    print(f.parts[l].size())

# print(f)

# Fproduct
output_1 = gelib.Fproduct(input_signal, f, hidden_maxl)

print("\nOutput after Fproduct ---------------------------------")
for l in range(len(output_1.parts)):
    print(output_1.parts[l].size())

print(output_1)

# Fmodsq
output_2 = gelib.Fmodsq(output_1, hidden_maxl)

print("\nOutput after Fmodsq -----------------------------------")
for l in range(len(output_2.parts)):
    print(output_2.parts[l].size())

# print(output_2)

print('Done')
