#!/usr/bin/python

from scipy.io import mmread
import numpy

adj = mmread("amazon0302_adj.mmio")
print type(adj)
csr = adj.tocsr()
numpy.savetxt("foo", csr.indptr, fmt = '%d')
