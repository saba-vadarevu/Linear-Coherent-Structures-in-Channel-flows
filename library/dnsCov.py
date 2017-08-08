"""
dnsCov.py
Read each DNS flowfield (spectral) in 'dataPath' defined below, 
extract "covariance" matrix for the velocity field for a bunch of Fourier modes,
average them over all the snapshots in 'dataPath'

IMPORTANT: 
    The covariance matrices include interpolation onto chebyshev (internal) nodes, and weighting

"""
import miscUtil
import numpy as np
import os
import pseudo
import sys
dataPath = os.environ['DATA186'] + 'spec/'
savePath = os.environ['DATA186'] + 'covN127/'
fNamePrefix = 'covR186N127'
currPath = os.getcwd()
os.chdir(dataPath)

uff = miscUtil.bin2arr(dataPath+'uFF_it100000.dat')
N = uff.shape[-1]
Lcov0 = 0; Lcov1 = 48 
Mcov0 = 0; Mcov1 = 32 
lArr = np.arange(2*Lcov1+1) 
lpArr = lArr.copy(); lpArr[Lcov1+1:] = lpArr[Lcov1+1:] - lpArr.size
mArr = np.arange(Mcov1+1) 
mpArr = mArr.copy()
#lArr = np.concatenate(( np.arange(Lcov0,Lcov1+1), np.arange(-Lcov1, -Lcov0+1) ))
#lpArr = np.concatenate(( np.arange(Lcov1-Lcov0+1),  ))

#==============
# Saving covariance matrices on-demand
lpArr = np.concatenate((np.array([0]),np.arange(5,20) ))
lArr = np.arange(lpArr.size)
mpArr = np.concatenate((np.array([0]),np.arange(5,20) ))
mArr = np.arange(mpArr.size)
#=============

# Code is incomplete. Get back to this later. 
Lx = 8.*np.pi; Ly = 3.*np.pi
a0 = 2.*np.pi/Lx; b0 = 2.*np.pi/Ly

# We'll be interpolating to a Chebyshev grid with 'nCheb' internal nodes
nCheb = 127 
covMat = np.zeros( (lpArr.size, mpArr.size, 3*nCheb, 3*nCheb), dtype=np.complex)

tRange = np.arange(100000, 150000,500)
uFiles = ['uFF_it%s.dat'%t for t in tRange]
vFiles = ['vFF_it%s.dat'%t for t in tRange]
wFiles = ['wFF_it%s.dat'%t for t in tRange]

U = np.zeros(N)
for fInd in range(len(uFiles)):
    print("loading ",uFiles[fInd])
    uff = miscUtil.bin2arr(uFiles[fInd])
    vff = miscUtil.bin2arr(vFiles[fInd])
    wff = miscUtil.bin2arr(wFiles[fInd])
  
    U[:] += np.real(uff[0,0])
   
    vel = np.concatenate(( uff, vff, wff), axis=2)    
    vel = miscUtil.interpDNS(vel.reshape((vel.size//N,N)) , nCheb=nCheb)[0]
    vel = vel.reshape((uff.shape[0], uff.shape[1], 3*nCheb,1))

    for ind0 in range(lpArr.size):
        lp = np.int(lpArr[ind0]); l = np.int(lArr[ind0])
        for ind1 in range(mpArr.size):
            mp = np.int(mpArr[ind1]); m = np.int(mArr[ind1])
            velVec = vel[lp,mp]
            covMat[l,m] += (velVec @ velVec.conj().T)

U = U/len(uFiles)
np.save('uMeanN192.npy',U)
covMat = covMat/len(uFiles)

# Weighting the covariance matrices with clencurt quadrature
weightDict = pseudo.weightMats(N=nCheb)
#weightsArr = pseudo.clencurt(nCheb)
#q = np.sqrt(weightsArr); q = np.concatenate(( q,q,q ))  # Tile them thrice for u,v,w
#Q = np.diag(q)      # Build a diagonal matrix out of it
W3s = weightDict['W3Sqrt']
for ind0 in range(covMat.shape[0]):
    lp = lpArr[ind0]
    for ind1 in range(covMat.shape[1]):
        mp = mpArr[ind1]
        covMatMode = W3s @ covMat[ind0, ind1] @ W3s
        fName = fNamePrefix + 'l%02dm%02d.npy'%(lp,mp)
        np.save(savePath+fName, covMatMode)
        print("Saved covariance for mode (%d,%d) to %s"%(lp,mp,fName))

os.chdir(currPath)


