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
savePath = os.environ['DATA186'] + 'cov/'
fNamePrefix = 'covR186'
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

# Code is incomplete. Get back to this later. 
Lx = 8.*np.pi; Ly = 3.*np.pi
a0 = 2.*np.pi/Lx; b0 = 2.*np.pi/Ly

# We'll be interpolating to a Chebyshev grid with 'nCheb' internal nodes
nCheb = 62
covMat = np.zeros( (lpArr.size, mpArr.size, 3*nCheb, 3*nCheb), dtype=np.complex)

tRange = np.arange(100000, 150000,500)
uFiles = ['uFF_it%s.dat'%t for t in tRange]
vFiles = ['vFF_it%s.dat'%t for t in tRange]
wFiles = ['wFF_it%s.dat'%t for t in tRange]

U = np.zeros(N)
for fInd in range(len(uFiles)):
    uff = miscUtil.bin2arr(uFiles[fInd])
    vff = miscUtil.bin2arr(vFiles[fInd])
    wff = miscUtil.bin2arr(wFiles[fInd])
  
    U[:] += np.real(uff[0,0])
   
    vel = np.concatenate(( uff, vff, wff), axis=2)    
    vel = miscUtil.interpDNS(vel.reshape((vel.size//N,N)) , nCheb=nCheb)[0]
    vel = vel.reshape((uff.shape[0], uff.shape[1], 3*nCheb,1))

    for lp in lpArr:
        for mp in mpArr:
            velVec = vel[lp,mp]
            covMat[lp,mp] += (velVec @ velVec.conj().T)

U = U/len(uFiles)
np.save('uMeanN192.npy',U)
covMat = covMat/len(uFiles)

# Weighting the covariance matrices with clencurt quadrature
weightsArr = pseudo.clencurt(nCheb)
q = np.sqrt(weightsArr); q = np.concatenate(( q,q,q ))  # Tile them thrice for u,v,w
Q = np.diag(q)      # Build a diagonal matrix out of it
for ind1 in range(covMat.shape[0]):
    for ind2 in range(covMat.shape[1]):
        covMat[ind1, ind2] = Q @ covMat[ind1, ind2] @ Q



# Save covariance matrix for each mode as a numpy binary
for lp in lpArr:
    for mp in mpArr:
        covMatMode = covMat[lp,mp]
        fName = fNamePrefix + 'N%dl%02dm%02d.npy'%(N,lp,mp)
        np.save(savePath+fName, covMatMode)
        print("Saved covariance for mode (%d,%d) to %s"%(lp,mp,fName))

os.chdir(currPath)


