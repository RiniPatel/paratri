#!/usr/bin/python

from scipy.io import mmread
import numpy

adj = mmread("amazon0302_adj.mmio")
# print type(adj)
csr = adj.tocsr()
numpy.savetxt("amazon0302_adj_IA.txt", csr.indptr, fmt = '%d')
numpy.savetxt("amazon0302_adj_JA.txt", csr.indices, fmt = '%d')
