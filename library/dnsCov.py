"""
dnsCov.py
Read each DNS flowfield (spectral) in 'dataPath' defined below, 
weight the velocity fields, 
extract covariance matrix for the velocity field for a bunch of Fourier modes,
average them over all the snapshots in 'dataPath'

"""
import miscUtil
import numpy as np
import os
import pseudo
import sys
dataPath = '/media/sabarish/channelData/R590/spec/'
savePath = '/media/sabarish/channelData/R590/cov/'
fNamePrefix = 'covR590'
currPath = os.getcwd()
os.chdir(dataPath)

uff = miscUtil.bin2arr(dataPath+'uFF_it50000.dat')
lArr = np.arange(uff.shape[0])
mArr = np.arange(uff.shape[1]) 
N = 384
Nlow = 64
interpFlag = True   #Interpolate onto low res grid
Lx = 2.*np.pi; Ly = np.pi
a0 = 2.*np.pi/Lx; b0 = 2.*np.pi/Ly

if interpFlag:
    w = pseudo.clencurt(Nlow+2)[1:-1]
    covMat = np.zeros( (lArr.size, mArr.size, 3*Nlow, 3*Nlow), dtype=np.complex)
else:
    w = pseudo.clencurt(N+2)[1:-1]
    covMat = np.zeros( (lArr.size, mArr.size, 3*N, 3*N), dtype=np.complex)
q = np.sqrt(w)
Q = np.diag( q.repeat(3) )  # Weight matrix for weighting vectors

tRange = np.arange(50000, 75050,50)
uFiles = ['uFF_it%s.dat'%t for t in tRange]
vFiles = ['vFF_it%s.dat'%t for t in tRange]
wFiles = ['wFF_it%s.dat'%t for t in tRange]

uffLow = np.zeros( (lArr.size, mArr.size, 1,N+2), dtype=np.complex )
vffLow = uffLow.copy(); wffLow = uffLow.copy()
velLow_1 = np.zeros( (lArr.size, mArr.size, 3, N+2), dtype=np.complex)

U = np.zeros(N)
Ulow = np.zeros(Nlow)
for fInd in range(len(uFiles)):
    uff = miscUtil.bin2arr(uFiles[fInd])
#sys.exit()
for fInd in range(len(uFiles)):
    uff = miscUtil.bin2arr(uFiles[fInd])
    vff = miscUtil.bin2arr(vFiles[fInd])
    wff = miscUtil.bin2arr(wFiles[fInd])
  
    # uff, vff, wff only have internal nodes. Need to extend with wall nodes for interpolation.
    # uffLow[:,:,:,1:-1] = uff[np.ix_(lArr,mArr)].reshape(( lArr.size, mArr.size, 1,N))
    # vffLow[:,:,:,1:-1] = vff[np.ix_(lArr,mArr)].reshape(( lArr.size, mArr.size, 1,N))
    # wffLow[:,:,:,1:-1] = wff[np.ix_(lArr,mArr)].reshape(( lArr.size, mArr.size, 1,N))
    velLow_1[:,:,0,1:-1] = uff
    velLow_1[:,:,1,1:-1] = vff
    velLow_1[:,:,2,1:-1] = wff
    U[:] += np.real(uff[0,0])
    if interpFlag:
        # Interpolating to low-res grid, 
        # going through chebcoeffs and then cutting them off instead of using barycentric thingy
        velChebCoeffs = pseudo.chebcoeffs( velLow_1 )   # Get chebcoeffs
        velChebCoeffs = velChebCoeffs[:,:,:, :Nlow+2]     # Get rid of higher coeffs
        velLow = pseudo.chebcoll_vec(velChebCoeffs)[:,:,:,1:-1]   # Back to collocation
        
    else:
        velLow = velLow_1[:,:,:,1:-1]
        Nlow = N
   
    velLow = velLow.reshape((lArr.size, mArr.size, 3*Nlow))

    for l in lArr:
        for m in mArr:
            velVecUnweighted = velLow[l,m]

            velVecWeighted = (Q @ velVecUnweighted).reshape((3*Nlow,1))
            covMat[l,m] += (velVecWeighted @ velVecWeighted.conj().T).reshape((3*Nlow,3*Nlow))

U = U/len(uFiles)
np.save('uMeanN384.npy',U)

covMat = covMat/len(uFiles)
assert covMat.shape== (lArr.size,mArr.size,3*Nlow, 3*Nlow)

# Save covariance matrix for each mode as a numpy binary
for l in lArr:
    if l <=96: lp = l
    else: lp = l-192
    for m in mArr:
        covMatMode = covMat[l,m]
        fName = fNamePrefix + 'N%dl%02dm%02d.npy'%(Nlow,lp,m)
        np.save(savePath+fName, covMatMode)
        print("Saved covariance for mode (%d,%d) to %s"%(lp,m,fName))




os.chdir(currPath)


