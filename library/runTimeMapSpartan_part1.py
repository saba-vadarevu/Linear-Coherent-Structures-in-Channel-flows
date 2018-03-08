import numpy as np
import scipy as sp
from scipy.io import loadmat, savemat
import impulseResponse as impres
import flowField 
import sys
import time
import os
import ops

Re = 13000.; N=401
tArr = np.arange(0.,1.01,0.025)
eddy = True
Lx = 4.*np.pi; Lz = np.pi
na = 160; nb = 96 
eps = 10.

#y0Arr = np.arange(-0.993, -0.95, 0.008)
y0Arr = np.arange(-0.993, -0.95, 0.008)[0:2]

a0 = 2.*np.pi/Lx; b0 = 2.*np.pi/Lz
aArr = a0*np.arange(-na//2, na//2)
bArr = b0*np.arange(nb+1)


fsAmp = np.zeros((9,3))
fsAmp[0] = np.array([1.,0.,0.])
fsAmp[1] = np.array([0.,1.,0.])
fsAmp[2] = np.array([0.,0.,1.])
fsAmp[3] = np.array([5.,1.,0.])
fsAmp[4] = np.array([-5.,1.,0.])
fsAmp[5] = np.array([1.,1.,0.])
fsAmp[6] = np.array([-1.,1.,0.])
fsAmp[7] = np.array([1.,1.,1.])
fsAmp[8] = np.array([-1.,1.,1.])



eArr = np.zeros((y0Arr.size, aArr.size, bArr.size, fsAmp.shape[0],3,tArr.size))
linInst = ops.linearize(N=N,Re=Re,turb=True)
for k in range(y0Arr.size):
    y0 = y0Arr[k]
    print("y0, y0plus: %.4g, %.4g"%(y0, Re*(1+y0)) )
    impulseArgs = {'N':N, 'Re':Re,'turb':True,'y0':y0,'eps':eps}
    for i0 in range(aArr.size):
        a = aArr[i0]
        if i0%5 == 0: print("a=",a)
        for i1 in range(bArr.size):
            b = bArr[i1]
            if (a==0.) and (b==0.):
                continue
            caseDict = impres.timeMap(a,b,tArr,linInst=linInst,\
                            fsAmp=fsAmp,eddy=eddy,\
                            coeffs=False, impulseArgs=impulseArgs.copy())
            energyArr = caseDict.pop('energyArr')
            eArr[k,i0,i1] = np.swapaxes(energyArr, 0,2)
            
    
        saveDict = {'aArr':aArr, 'bArr': bArr, 'tArr':tArr, 'y0Arr':y0Arr,\
                'turb':True, 'eddy':eddy, 'Re':Re, 'N':N, 'fsAmp':fsAmp,\
                'energyArr':eArr,'k':k, 'i0':i0,'i1':i1, 'eps':eps}
        saveDict.update(caseDict)
        
    savemat("/kepler/sabarish/impulseResponse/energyEddyRe13000NearWall_part1.mat",saveDict)


