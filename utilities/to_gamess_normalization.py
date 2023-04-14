#!/usr/bin/env python

from resultsFile.lib.basis import xyz_from_lm
from trexio_tools.converters import cart_sphe
import numpy as np


def run(l):
  xyz_from_lm_results = [[0]+ list(xyz_from_lm(l,0))]
  for m in range(1,l+1):
     xyz_from_lm_results.append( [2*m-1] + list(xyz_from_lm(l,m)) )
     xyz_from_lm_results.append( [2*m] + list(xyz_from_lm(l,-m)) )


  # Find the ordering of xyz functions
  order = {}
  for _, p, _ in xyz_from_lm_results:
     for x in p:
       order[x] = 0
  lst = [ x for x in order.keys()]
  lst.sort()
  lst.reverse()
  for i, p in enumerate(lst):
     order[p] = i

  ref = cart_sphe.data[l]
  new = np.array(cart_sphe.data[l])
  new -= new
  for i, p, c in xyz_from_lm_results:
     for p, c in zip(p,c):
       new[order[p],i] = c

  new = new/(ref+1.e-32)
  res = np.array(new[:,0])
  for i in range(new.shape[0]):
    for j in range(new.shape[1]):
      res[i] = max(res[i], new[i,j])
  return list(res)


for l in range(11):
  print(l, run(l))
