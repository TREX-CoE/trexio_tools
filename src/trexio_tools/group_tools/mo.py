#!/usr/bin/env python3

import trexio
import numpy as np
from . import ao as trexio_ao

def read(trexio_file):
    r = {}

    r["ao"]     = trexio_ao.read(trexio_file)
    r["num"]    = trexio.read_mo_num(trexio_file)
    r["coefficient"]  = trexio.read_mo_coefficient(trexio_file)

#    if trexio.has_mo_type(trexio_file):
#        r["type"]   = trexio.read_mo_type(trexio_file)
#    if trexio.has_mo_class(trexio_file):
#        r["class"]   = trexio.read_mo_class(trexio_file)
#    if trexio.has_mo_symmetry(trexio_file):
#        r["symmetry"]   = trexio.read_mo_symmetry(trexio_file)
#    if trexio.has_mo_occupation(trexio_file):
#        r["occupation"]   = trexio.read_mo_occupation(trexio_file)

    return r


def value(mo, r):
    """
    Evaluates all the MOs at R=(x,y,z)
    """

    ao = mo["ao"]
    ao_value = trexio_ao.value(ao, r)
    coef = mo["coefficient"]
    return coef @ ao_value
