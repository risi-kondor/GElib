# WE SHOULD REMOVE THE import *  !!!
import torch

from gelib_base import SO3element 
from gelib_base import SO3type 
from gelib_base import SO3bitype 

from SO3part import *
from SO3weights import *
from SO3weightsArr import *
from SO3vec import *
from SO3weights import *
from SO3mvec import *
from SO3partArr import *
from SO3vecArr import *
from SO3weightsArr import *
from Wigner import *
from CGtensor import *

from SO3partB import *
from SO3vecB import *

from SO3partC import *
from SO3vecC import *
from SO3partArrC import *
from SO3vecArrC import *

from SO3bipart import *
from SO3bipartArr import *
from SO3bivec import *
from SO3bivecArr import *


def CGproduct(x, y, maxl=-1):
    return x.CGproduct(y, maxl)

def DiagCGproduct(x, y, maxl=-1):
    return x.DiagCGproduct(y, maxl)

def CGtransform(x,l=-1):
    return x.CGtransform(l)

