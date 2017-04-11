import numpy as np
from warnings import warn


def bin2arr(fName, L=96,M=64,N=384):
    """ Read a binary with spectral fields and return a numpy array
    Inputs:
        L,M (int): Number of Fourier modes in 
    """
    with open(fName,'rb') as inFile:
        specArr = np.fromfile(fName,dtype=np.float,count=-1)
        if specArr.size%(L*M) :
            print("Sizes don't match.. binary has %d elements, and L and M are %d and %d"%(specArr.size, L, M))
        specArr = specArr.reshape((2*L+1,M, specArr.size//((2*L+1)*M)))
        Nz2 = specArr.shape[2]; Nz = Nz2//2
        if Nz != N: warn("The supplied N is not consistent with the size of the array.")
        specArrComp = specArr[:,:,:Nz] + 1.j * specArr[:,:,Nz:]

    return specArrComp

def phys2onePointSpec(t=1000,Nx=384,Ny=256,Nz=384, L=32,M=16,saveField=True,loadPrefix='./', savePrefix='./', **kwargs):
    """ Read binary files with physical fields and save/return spectra of u_i*u_i
    IMPORTANT: The returned/saved fields are not time-averaged. 
    Inputs:
        t (int): Time-snapshot
        L, M (int): Number of streamwise and spanwise Fourier modes to keep 
        saveField (bool=True): If True, print the spectral field. If False, return array
        prefix (str = './') : Use this if physical field files are at a different location.
        The u-field is expected to have the file name    prefix+'u_it'+str(t)+'.dat'
    Outputs: 
        If saveField, prints files uu_it%s.dat %t, similarly for uv, vv, etc..
        If not saveField, returns a numpy.ndarray of shape (9,L,M,Nz)
    """
    uFile = loadPrefix+'u_it%s.dat'%t
    vFile = loadPrefix+'v_it%s.dat'%t
    wFile = loadPrefix+'w_it%s.dat'%t

    velPhys = np.zeros((3,Nx,Ny,Nz),dtype=np.float)

    with open(uFile, 'rb') as inFile:
        uPhys = np.fromfile(inFile, dtype=np.float, count=-1)
        velPhys[0] = uPhys.reshape((Nx,Ny,Nz))
    with open(vFile, 'rb') as inFile:
        vPhys = np.fromfile(inFile, dtype=np.float, count=-1)
        velPhys[1] = vPhys.reshape((Nx,Ny,Nz))
    with open(wFile, 'rb') as inFile:
        wPhys = np.fromfile(inFile, dtype=np.float, count=-1)
        velPhys[2] = wPhys.reshape((Nx,Ny,Nz))
    del uPhys, vPhys, wPhys

    meanVel = np.mean( np.mean( velPhys, axis=1), axis=1).reshape((3,1,1,Nz))

    # Subtract the mean (spatially averaged) velocity
    velPhys = velPhys - meanVel

    enerPhys = velPhys.reshape((1,3,Nx,Ny,Nz)) * velPhys.reshape((3,1,Nx,Ny,Nz))
    del velPhys
    
    # FFT of enerPhys:
    enerSpec = np.zeros((3,3,2*L+1, M, Nz))     
    # I keep only M instead of M+1 because I made this mistake in an earlier code,
    #   and am now keeping it for consistency

    enerSpec = (np.fft.rfftn( enerPhys, axes=(2,3))/Nx/Ny)
    # real FFT of physical fields for Reynolds stresses
    
    enerSpec = enerSpec[:, :, np.r_[:L+1, Nx-1:Nx-L-1:-1], :M]
    # Keeping only -L < l <= L and  m <M modes for arguments L and M to the function

    if not saveField:
        return enerSpec
    else:
        saveFileTemp = '_it%s.dat'%t
        enerSpecReal = np.concatenate( (np.real(enerSpec), np.imag(enerSpec)), axis=4)

        outNameList = np.array(['uu','uv','uw', 'vu', 'vv', 'vw', 'wu','wv','ww']).reshape((3,3))
        for i1 in range(3):
            for i2 in range(3):
                outName = savePrefix + outNameList[i1,i2]+'_it%s.dat'%t
                with open(outName,'wb') as outFile:
                    enerSpecReal[i1,i2].tofile(outFile)
                print("Wrote %s 2-d spectra at time %d to %s"%(outNameList[i1,i2],t,outName))
        return None




