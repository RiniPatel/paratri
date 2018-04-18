#!/usr/bin/python

from scipy.io import mmread
import numpy

import os
for file in os.listdir("./"):
    if file.endswith(".mmio"):
        adj = mmread(file)
        csr = adj.tocsr()
        file = os.path.splitext(file)[0]
        print file
        numpy.savetxt(file + "_IA.txt", csr.indptr, fmt = '%d')
        numpy.savetxt(file + "_JA.txt", csr.indices, fmt = '%d')
        print "Generated " + file + "_IA.txt and " + file + "_JA.txt"
