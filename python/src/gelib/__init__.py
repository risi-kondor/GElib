# WE SHOULD REMOVE THE import *  !!!
import torch

#from gelib_base import SO3element 
#from gelib_base import SO3type 
#from gelib_base import SO3bitype 

from gelib.gelib_common import *
#from gelib.gather_map import *
# Removed redundant import of gather_map

from gelib.SO3element import *
from gelib.SO3irrep import *
from gelib.SO3part import *
from gelib.SO3vec import *

from gelib.SO3type import *
# from gelib.SO3weights import *
# from gelib.SO3weightsArr import *
# from gelib.SO3weights import *
# from gelib.SO3mvec import *

from gelib.SO3partArr import *
from gelib.SO3vecArr import *

#from gelib.SO3weightsArr import *
#from gelib.Wigner import *
#from gelib.CGtensor import *

#from gelib.SO3partB import *
#from gelib.SO3vecB import *

#from gelib.SO3partC import *
#from gelib.SO3vecC import *
#from gelib.SO3partArrC import *
#from gelib.SO3vecArrC import *

#from gelib.SO3bipart import *
#from gelib.SO3bipartArr import *
#from gelib.SO3bivec import *
#from gelib.SO3bivecArr import *
# from gelib.SO3weightsArr import *
# from gelib.Wigner import *
# from gelib.CGtensor import *


def CGproduct(x, y, maxl=-1):
    return x.CGproduct(y, maxl)

def DiagCGproduct(x, y, maxl=-1):
    return x.DiagCGproduct(y, maxl)

def CGtransform(x,l=-1):
    return x.CGtransform(l)

def gather(x,gmap,dim=0):
    return x.gather(gmap,dim)

