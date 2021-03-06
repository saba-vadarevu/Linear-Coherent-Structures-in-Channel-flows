import numpy as np
from warnings import warn
import sys
import os
import os.path
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pseudo
import sys


def _floatGCD(a,b,reltol=1.e-05, abstol=1.e-09):
    t = min(a, b)
    while b > reltol*t + abstol:
        a, b = b, a%b
    return a

def _arrGCD(arr, tol = 1.e-09 ):
    arr = np.array([arr]).flatten()
    assert (arr > 0.).all()
    if np.linalg.norm( ( arr + tol ) % arr[0] ) < 10.* arr.size * tol:
        return arr[0]
    else :
        gcdEst = arr[0] 
        for ind in range(1, arr.size):
            gcdEst = _floatGCD( gcdEst, arr[ind] )
        return gcdEst
        
def _areSame(arr1,arr2,tol=1.e-09):
    return ( np.linalg.norm(arr1 - arr2) <= tol )

def _nearestEntry(arr,a0):
    return arr.flat[np.abs(arr-a0).argmin()]

def _nearestInd(arr,a0):
    return np.abs(arr-a0).argmin()



def phys2spec(t=100000, L=64,M=48,Nx=512,Ny=320,Nz=192,loadPath=None,savePath=None,prefixes=['u','v','w']):
    """ Read binary file for physical field and write binary file for spectral field
    Inputs:
        t (int=100000):    Index for time snapshot
        Nx(int=512):    Number of streamwise nodes in physical data
        Ny(int=320):    Number of spanwise nodes in physical data
        Nz(int=192):    Number of wall-normal nodes in physical data. These are cell-centred, not the usual Cheb nodes
        loadPath(str='$DATA/phys/'): Path where physical binaries are found. Uses system environment variable DATA if not supplied
        savePath(str='$DATA/spec/'): Path to save spectral binaries to
        prefixes(list=['u','v','w']):   Fields to read/write at the specified time
        L (int=96):     Max streamwise Fourier mode to write (write coefficients for -L < l <= L)
        M (int=64):     Max spanwise Fourier mdoe to write (write coefficients for 0<= m <= M)
    Outputs:
        None        (Fields written to .dat files)
        """
    assert L < Nx//2, "Do not save all wavenumbers."
    if loadPath is None:
        loadPath = os.environ['DATA'] + 'phys/'
    if savePath is None:
        savePath = os.environ['DATA'] + 'spec/'

    for pfix in prefixes:
        fName = loadPath+ pfix + '_it%s.dat'%t
        xRange = np.r_[0:L+1,Nx-L:Nx]
        #try:
        with open(fName,'rb') as inFile:
            uArr = np.fromfile(inFile, dtype=np.float, count=-1)
        uArr = uArr.reshape((Nx,Ny,Nz))
        uArrFFcomplex = np.fft.rfftn(uArr, axes=(0,1))/Nx/Ny
        uArrFFcomplex = uArrFFcomplex[ xRange, :M+1 ]
        uArrFFreal = np.concatenate( (np.real(uArrFFcomplex), np.imag(uArrFFcomplex)), axis=2)
        newName = savePath+ pfix + 'FF_it%s.dat'%t

        with open(newName,'wb') as outFile:
            uArrFFreal.tofile(outFile)
        print("Successfully FFTd %s to %s" %(fName,newName))
        #except:
            #print("Could not FFT %s for whatever reason.."%fName)
        sys.stdout.flush()
    return

def bin2arr(fName, L=64,M=48,N=192):
    """ Read a binary with spectral fields and return a numpy array
    Inputs:
        L,M (int=64,48): Number of positive Fourier modes
                    Actual modes go as -L=<l<=L, 0<=m<=M, so there's (2L)*(M+1) modes in total
        N (int=192):    Number of cells in DNS. Nodes are not Cheb nodes, but centers of these cells whose edges are cheb nodes
    Outputs:
        specArr: Complex array of Fourier coefficients of shape (2L+1, M+1, N); read from a binary with float entries
    """
    with open(fName,'rb') as inFile:
        specArr = np.fromfile(fName,dtype=np.float,count=-1)
        if specArr.size%((2*L+1)*(M+1)) :
            print("Sizes don't match.. binary has %d elements, and 2L+1 and M+1 are %d and %d"%(specArr.size, 2*L+1, M+1))
        specArr = specArr.reshape((2*L+1,M+1, specArr.size//((2*L+1)*(M+1))))
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
    
    enerSpec = enerSpec[:, :, np.r_[:L+1, Nx-L:Nx], :M]
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



def nodesCellCenters(nCells=192,**kwargs):
    """ Returns locations of cell-centers when edges of the cells are defined by Chebyshev nodes.
    Inputs: 
        nCells (=192): Number of cells. Keyword argument used to avoid ambiguity with number of faces.
        Optional kwargs:
            nGhost (=0):    Number of ghost cells to pad internal cells with
            Lz (=2.):       Wall-normal extent to scale with. Don't worry about this now.
    Outputs:
        zCC: Cell-centers for internal cells"""
    nz = nCells; 
    nGhost = kwargs.get('nGhost',0)     # If nGhost is supplied, pad on either side 
    Lz = kwargs.get('Lz',2.)
    indArr = np.arange(nz)
    # Defining cells: 
    ztmp1 = np.cos(np.pi * indArr/nz)   # Top edge of internal cell, cos(n*pi/N) for n=0 to N-1
    ztmp2 = np.cos(np.pi * (indArr+1)/nz)   # Bottom edge of internal cell, cos(n*pi/N) for n=1 to N
    # z-coords for cell centers: zCC
    zCC = np.zeros(nz + 2*nGhost)
    zCC[nGhost:zCC.size-nGhost] = (ztmp1 + ztmp2)/2.
    # Size of ghost cells are symmetric about the walls on either end
    #zCC[nGhost-1::-1] = 1. + (1.-zCC[:nGhost])
    #zCC[:nz+nGhost-1:-1] = -1. - (1.+zCC[nz:nz+nGhost])  # Size of ghost cells at top wall
    zCC = Lz/2.* zCC

    # Return inverted list so that it goes from -1 to 1 instead of 1 to -1,
    #   because the DNS data is saved from 0 to 2 (i.e., increasing y)
    return zCC[::-1]

def cenDiff(arr, nCells=192,**kwargs):
    """ Returns differentiation matrix for (1st order accurate) central difference on cell-center nodes
        (see nodesCellCenters)
        Inputs:
            arr : Scalar data on cell-center nodes 
            nCells (=192): Number of cells. 
                        Currently, ghost cells are not supported. 
            kwargs:
                bvals: list/array/ordered pair with bvals at 1 and -1
                    Default is (0,0)
        Outputs:
            dArr: First derivative of the array supplied
    """
    # My code is presently built to handle data only on internal nodes, i.e. without walls
    # This differentiation matrix supposes that whatever field it is we're differentiating is zero at the walls
    # To allow for easy central differencing without any mess, I'm adding nodes for the walls

    # The derivative is calculated like so:
    # For function f with values fi on grid points yi, 
    # Then, f'i is given as
    #   f'i =   (yi-yim1)/(yip1-yim1) * (fip1-fi)/(yip1-yi)  + (yip1-yi)/(yip1-yim1) * (fi-fim1)/yi-yim1)

    # Refer to http://cfd.mace.manchester.ac.uk/twiki/pub/Main/TimCraftNotes_All_Access/cfd1-findiffs.pdf for derivation, which can be expressed in simpler terms as above
    assert arr.size == nCells
    bvals = kwargs.get('bvals', [0.,0.])
    f = np.zeros(nCells+2, dtype=arr.dtype)
    f[1:-1] = arr
    f[0] = bvals[0]; f[-1] = bvals[-1]

    yCC = np.zeros(nCells+2)
    yCC[1:-1] = nodesCellCenters(nCells=nCells, nGhost=0, Lz=2.)
    yCC[-1] = 2.
    # yCC from nodesCellCenters goes from 0 to 2 (both exclusive) instead of my usual convention of -1 to 1. 
    # Actually, my convention is for y to go from 1 to -1, not -1 to 1. To reflect this:
    yCC = yCC[::-1]
    warn("arr supplied to cenDiff must be on y from 1 to -1, NOT -1 to 1")

    ym1 = yCC[:-2]
    y0  = yCC[1:-1]
    yp1 = yCC[2:]

    fm1 = f[:-2]
    f0  = f[1:-1]
    fp1 = f[2:]

    df = (y0-ym1)/(yp1-ym1) * (fp1-f0)/(yp1-y0)  +  (yp1-y0)/(yp1-ym1) * (f0-fm1)/(y0-ym1)
    
    return  df 



def interpDNS(ccArr,**kwargs):
    """
    Interpolate data from DNS on cell-centers to the Chebyshev nodes. Scipy's cubic splines are used.
    WARNING: This function assumes that the array has a value zero at the boundary.
    Inputs:
        ccArr:  Data array on cell centers. Last dimension of ccArr denotes cell-center nodes
        nCheb (optional): Number of internal chebyshev nodes to interpolate to.
                Defaults to ccArr.shape[-1] - 1
    Outputs:
        chebArr:    Data array on Chebyshev nodes
        """
    origDim = ccArr.ndim
    if origDim > 2:
        warn("interpDNS only returns 1d or 2d arrays. If the input array has more dimensions, remember to reshape it afterwards.")
    N = ccArr.shape[-1]; ccZ = nodesCellCenters(nCells=N)   # Number of cell-centers, and their locations in (1,-1)
    ccArr = ccArr.reshape((ccArr.size//N,N))

    walledArr = np.zeros((ccArr.shape[0], N+2),dtype=ccArr.dtype); walledZ= np.zeros(N+2)  # Extend nodes and values to include z=1,-1
    walledArr[:,1:-1] = ccArr[:] 
    walledZ[1:-1] = ccZ; walledZ[0] = 1.; walledZ[-1] = -1.    # Data is 0 at either walls

    # Defining nodes to interpolate to, and an array to save this data
    nCheb = kwargs.get('nCheb',N-1) 
    chebZ = pseudo.chebdif(nCheb,1)[0]    # N cell centers means N+1 edges including walls
    chebArr = np.zeros((ccArr.shape[0], chebZ.size),dtype=ccArr.dtype)

    for ind in range(ccArr.shape[0]):
        intFun = interp1d(ccZ, ccArr[ind], kind='cubic')    # Cubic spline interpolation function for each scalar in ccArr
        chebArr[ind] = intFun(chebZ)    # Interpolate to chebyshev nodes

    if origDim == 1: chebArr = chebArr.flatten()

    return  chebArr, chebZ





def DNSderivs(U,**kwargs):
    """ 
    Obsolete. Use semi-empirical equation for U instead.
    Derivatives on the z-grid used in Daniel's DNS code, using the same scheme used in the DNS.
    Following the function helmholtz1d_operator_u4 in diffuseu.c
    Inputs:
        U: Some scalar on 'z', usually the mean velocity averaged in space and/or time
        """
    a1 = 1.125
    a2 = -0.125/3.
    # Some coefficients of the differentiation scheme of Daniel's code
    nGhost = 3  # Number of ghost cells each beyond the top and bottom walls
    warn('Ensure that the inputs, U and z, include 3 ghost cells along wall-normal on either side')
    warn("Test this function with simple cubic profile on z")

    # Defining the z-grid of Daniel's code, see lines 1119-1120 in chanfast.c
    Lz = kwargs.get('Lz',2.)
    nz = U.size - 2*nGhost  
    indArr = np.arange(nz)
    # Defining cells: 
    ztmp1 = np.cos(np.pi * indArr/nz)   # Top edge of internal cell, cos(n*pi/N) for n=0 to N-1
    ztmp2 = np.cos(np.pi * (indArr+1)/nz)   # Bottom edge of internal cell, cos(n*pi/N) for n=1 to N
    dztmp = Lz/2.*( ztmp1 - ztmp2 )        # Size of each internal cell, scaled to Lz/2
    dz = np.zeros(U.size)               # Size of cells, including ghost cells
    dz[nGhost:dz.size-nGhost] = dztmp   # Size of internal cells
    dz[nGhost-1::-1] = dz[nGhost:2*nGhost]  # Size of ghost cells at bottom wall
    dz[:nz+nGhost-1:-1] = dz[nz:nz+nGhost]  # Size of ghost cells at top wall
    # dz is symmetric about centerline, so don't worry about top/bottom wall remarks

    # z-coords for cell centers: zCC
    zCC = np.zeros(dz.size)
    zCC[nGhost:-nGhost] = (ztmp1 + ztmp2)/2.
    zCC[nGhost-1::-1] = 1. + (1.-zCC[:nGhost])
    zCC[:nz+nGhost-1:-1] = -1. - (1.+zCC[nz:nz+nGhost])  # Size of ghost cells at top wall
    zCC = Lz/2.* zCC

    Uint = U[nGhost: U.size-nGhost]     # Scalar at internal nodes: 'k'
    Up1 = U[nGhost+1:U.size-nGhost+1]   # Nodes at k+1
    Up2 = U[nGhost+2:U.size-nGhost+2]   # Nodes at k+2, and so on..
    Up3 = U[nGhost+3:U.size-nGhost+3]
    Um1 = U[nGhost-1:U.size-nGhost-1]
    Um2 = U[nGhost-2:U.size-nGhost-2]
    Um3 = U[nGhost-3:U.size-nGhost-3]

    dzint = dz[nGhost: U.size-nGhost]     # Cell size at internal nodes: 'k'
    dzp1 = dz[nGhost+1:U.size-nGhost+1]   # Nodes at k+1
    dzp2 = dz[nGhost+2:U.size-nGhost+2]   # Nodes at k+2, and so on..
    dzm1 = dz[nGhost-1:U.size-nGhost-1]
    dzm2 = dz[nGhost-2:U.size-nGhost-2]
    

    # It's probably just the usual 4th order central difference, but does it matter?
    # I'm using the exact same expressions used by Daniel in his DNS
    wp3h = a1*( Up2 - Up1 )  +  a2*( Up3 - Uint )
    wp1h = a1*( Up1 - Uint)  +  a2*( Up2 - Um1  )
    wm1h = a1*( Uint- Um1 )  +  a2*( Up1 - Um2  )
    wm3h = a1*( Um1 - Um2 )  +  a2*( Uint- Um3  )

    dzp3h = 0.5*( dzp2 + dzp1 )
    dzp1h = 0.5*( dzp1 + dzint)
    dzm1h = 0.5*( dzm1 + dzint)
    dzm3h = 0.5*( dzm2 + dzm1 )

    d2Uint = (a1*( wp1h/dzp1h - wm1h/dzm1h ) + a2*( wp3h/dzp3h - wm3h/dzm3h ) )/dzint


    # First derivative: The details are a bit shady for now, but I'll worry about it depending on how the tests go


    return d2Uint



def binarySearch(someFun, minVar, maxVar, tol=1.0e-03, nSteps=20, **kwargs):
    """ Find var satisfying  |someFun(var, **kwargs)| <= tol, for var in [minVar,maxVar]
    Exit with warning (print) if someFun does not change sign from minVar to maxVar
    Inputs:
        someFun: Function that takes exactly one positional argument. Keyword arguments are allwed.
        minVar :  Lower bound on variable for binary search
        maxVar :  Upper bound on variable for binary search
        tol (=0.001):  Tolerance for the zero of someFun()
        nSteps (=20):  Number of steps for binary search
        **kwargs:      Keyword arguments to pass to someFun()
    Outputs:
        newVar:     Variable at which abs value of someFun() goes below tolerance
    """
    newFun = lambda var: someFun(var,**kwargs)
    tolMet = lambda var: np.abs(var) <= tol
    signChange = lambda var1,var2: newFun(var1) * newFun(var2) < 0.
    # If tolerance is met at either minVar or maxVar, don't do the search
    if tolMet(newFun(minVar)): 
        print("tolerance %.3g met at minVar=%.3g. Returning..."%(tol,minVar))
        return minVar
    if tolMet(newFun(maxVar)): 
        print("tolerance %.3g met at maxVar=%.3g. Returning..."%(tol,maxVar))
        return maxVar

    assert minVar < maxVar
    assert signChange(minVar, maxVar)
    for n in range(nSteps):
        newVar = (minVar + maxVar)/2.
        if tolMet(newFun(newVar)): return newVar
        if signChange(minVar, newVar):
            maxVar = newVar
            continue
        elif signChange(newVar, maxVar):
            minVar = newVar
            continue
        else:
            print("Something's not right in the binary search")
            print("minVar, newVar, maxVar are", minVar, newVar, maxVar)
            print("function evals at these values are", newFun(minVar), newFun(newVar), newFun(maxVar))
            print("returning newVar...")
            return newVar
    print("max steps reached in binary search. Returning..")
    return newVar
                                            


def getPointData(loadDir, tArr, xyzArr, nx=2048, ny=512, nz=1152):
    """ Get instantaneous u,v,w data at a few points in the box for a range of times 
    Input:
        loadDir: Path to outputdir of the DNS data
        tArr: List/np.ndarray of times (int) 
        xyzArr: numpy array of shape (m,3) specifying indices for x, y, z
                    Remember, x and z are uniformly spaced, y is a Chebyshev grid
        nx (=2048): Number of streamwise nodes used in DNS
        ny (=512):  Number of wall-normal nodes used in DNS
        nz (=1152): Number of spanwise nodes used in DNS
                    The defaults work for ReTau=1000
    Output:
        outDict with keys 
            'tArr', 'xyzArr', 'uArr', 'vArr, 'wArr' 
                The last 3 are of shape (m,tArr.size)
    """
    if not loadDir.endswith('/'): loadDir = loadDir + '/'
    tArr = np.array(tArr).flatten()
    assert os.path.isfile(loadDir + 'u_it%d.dat'%tArr[0]), "Seems the loadDir (%s) is wrong"%loadDir
    xyzArr = np.array(xyzArr)
    xyzArr = xyzArr.reshape((xyzArr.size//3, 3))

    uArr = np.zeros((xyzArr.shape[0], tArr.size)); vArr = uArr.copy(); wArr = uArr.copy()
    loadFull = True 
    for i1 in range(tArr.size):
        t = tArr[i1]
        with open(loadDir+"u_it%d.dat"%t, 'rb') as inFile:
            if loadFull:
                tmpArr = np.fromfile(inFile, dtype=np.float, count=-1)
                tmpArr = tmpArr.reshape((nx,nz,ny))
            
            for i0 in range(xyzArr.shape[0]):
                xyzInd = xyzArr[i0]; 
                if not loadFull:
                    # Remember that DNS data goes as x,z,y
                    nByte = 8*( (ny*nz)*xyzInd[0] + (ny)*xyzInd[2] + xyzInd[1] )
                    inFile.seek(nByte,1)    # Move 'nByte' bytes from the first byte
                    uArr[i0, i1] = float(inFile.read(8))   # Read 8 bytes (double precision float)
                else :
                    uArr[i0,i1] = tmpArr[xyzInd[0], xyzInd[2], xyzInd[1]]
        with open(loadDir+"v_it%d.dat"%t, 'rb') as inFile:
            if loadFull:
                tmpArr = np.fromfile(inFile, dtype=np.float, count=-1)
                tmpArr = tmpArr.reshape((nx,nz,ny))
            
            for i0 in range(xyzArr.shape[0]):
                xyzInd = xyzArr[i0]; 
                if not loadFull:
                    # Remember that DNS data goes as x,z,y
                    nByte = 8*( (ny*nz)*xyzInd[0] + (ny)*xyzInd[2] + xyzInd[1] )
                    inFile.seek(nByte,1)    # Move 'nByte' bytes from the first byte
                    vArr[i0, i1] = float(inFile.read(8))   # Read 8 bytes (double precision float)
                else :
                    vArr[i0,i1] = tmpArr[xyzInd[0], xyzInd[2], xyzInd[1]]
        with open(loadDir+"w_it%d.dat"%t, 'rb') as inFile:
            if loadFull:
                tmpArr = np.fromfile(inFile, dtype=np.float, count=-1)
                tmpArr = tmpArr.reshape((nx,nz,ny))
            
            for i0 in range(xyzArr.shape[0]):
                xyzInd = xyzArr[i0]; 
                if not loadFull:
                    # Remember that DNS data goes as x,z,y
                    nByte = 8*( (ny*nz)*xyzInd[0] + (ny)*xyzInd[2] + xyzInd[1] )
                    inFile.seek(nByte,1)    # Move 'nByte' bytes from the first byte
                    wArr[i0, i1] = float(inFile.read(8))   # Read 8 bytes (double precision float)
                else :
                    wArr[i0,i1] = tmpArr[xyzInd[0], xyzInd[2], xyzInd[1]]
       
    return {'tArr':tArr, 'xyzArr':xyzArr, 'uArr':uArr, 'vArr':vArr, 'wArr':wArr, 'loadDir':loadDir}
