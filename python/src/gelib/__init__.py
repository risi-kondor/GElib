# WE SHOULD REMOVE THE import *  !!!
import torch

#from gelib_base import SO3element 
#from gelib_base import SO3type 
#from gelib_base import SO3bitype 

from gelib.SO3type import *
from gelib.SO3part import *
from gelib.SO3vec import *
#from gelib.SO3weights import *
#from gelib.SO3weightsArr import *
# from gelib.SO3weights import *
# from gelib.SO3mvec import *
from gelib.SO3partArr import *
from gelib.SO3vecArr import *
# from gelib.SO3weightsArr import *
# from gelib.Wigner import *
# from gelib.CGtensor import *


def CGproduct(x, y, maxl=-1):
    return x.CGproduct(y, maxl)

def DiagCGproduct(x, y, maxl=-1):
    return x.DiagCGproduct(y, maxl)

def CGtransform(x,l=-1):
    return x.CGtransform(l)

