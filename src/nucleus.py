#!/usr/bin/env python3

import trexio
import numpy as np

def read(trexio_file):
    r = {}

    r["num"] =  trexio.read_nucleus_num(trexio_file)
    r["charge"] =  trexio.read_nucleus_charge(trexio_file)
    r["coord"] =  trexio.read_nucleus_coord(trexio_file)
    r["label"] =  trexio.read_nucleus_label(trexio_file)

    return r



