#!/usr/bin/python2
from __future__ import print_function
## ==============================================
# npy2mat
# Convert .npy binaries to .mat binaries
# Should work with python2
## ==============================================

import numpy as np
from scipy.io import savemat
import argparse
import glob
import os

parser = argparse.ArgumentParser(description=\
        "Reads numpy binaries (with complex arrays) and writes to .mat files.\
         Takes as arguments an input file name and an output file name.")
parser.add_argument("inFile", help="Input file name (mandatory)", type=str)
parser.add_argument("--inPath", help="Directory path of input file (optional, default=./)", default=os.getcwd(), type=str)
parser.add_argument("--outFile", help="Output file name (optional, default is the same as inFile, but with .mat suffix", type=str)
parser.add_argument("--outPath", help="Directory path for output file (optional, default=inPath)", default=None, type=str)

args = parser.parse_args()
inPath = args.inPath
inFile = args.inFile

# If inFile doesn't end with .npy, get chars before '.' in inFile and add .npy
if not inFile.endswith('.npy'):
    inFile = inFile.split('.')[0]
    inFile = inFile + '.npy'
if not inPath.endswith('/'): 
    print("appending '/' to inPath")
    inPath = inPath + '/'

if args.outFile is not None: outFile = args.outFile
else: outFile = inFile
# If outFile doesn't end with .mat, get chars before '.' in inFile and add .mat
if not outFile.endswith('.mat'):
    outFile = outFile.split('.')[0]
    outFile = outFile+'.mat'
if args.outPath is not None: outPath = args.outPath
else: outPath = inPath
if not outPath.endswith('/'): outPath = outPath + '/'

inArr = np.load(inPath + inFile)
savemat(outPath+outFile, {'covMat':inArr})
print("Saved covariance matrix from %s to %s"%(inPath+inFile, outPath+outFile))
