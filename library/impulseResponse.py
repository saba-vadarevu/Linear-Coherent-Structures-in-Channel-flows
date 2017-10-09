# impulseResponse.py
""" 
Define functions that take as parameters the Fourier wavenumbers, time, direction and amplitude of impulse, 
and return the Fourier coefficients or energy at that particular time.
Also, H2norms for inifinite time and finite time horizons.
"""
import numpy as np
import ops
import pseudo
from warnings import warn
from scipy.linalg import expm, solve_sylvester

def impulseVec(a,b,**kwargs):
    """ Generate 3C impulse vector Fs (see sec. 10.1 of Jovanovic's thesis) from wall=normal profile 'fs' 
    Inputs:
        a,b:    Streamwise and spanwise wavenumbers 
        kwargs with keys 
            fs:     Wall-normal profile for impulse; if not supplied, build using keys
            y0, eps, N: location of impulse, wall-normal size of impulse, size of Chebysev collocation vector. Defaults are -0.9, 1/2000, 251
            DeltaInv: 
                    Inverse of the matrix (D2 - (a^2+b^2)I)
    Outputs:
        impulseDict with keys
            Fs_4x2N, Fsadj_4x2N, a, b, DeltaInv
            (computing Fs isn't very expensive. So ALWAYS compute for x,y,z,xyz)

    """
    # Need fs, DeltaInv, and impulseDir to compute Fs and FsAdj
    # Compute fs if not supplied:
    if 'fs' not in kwargs:
        warn("fs not supplied. Building it using y0, eps...")
        if 'y0' not in kwargs: y0 = -0.9; warn("y0 not supplied. Using -0.9...")
        else: y0 = kwargs['y0']
        if 'eps' not in kwargs: eps = 1./2000.; warn("eps not supplied. Using 1/2000...")
        else: eps = kwargs['eps']
        if 'N' not in kwargs: N = 251; warn("N not supplied. Using 251...")
        else: N = kwargs['N']

        y, DM = pseudo.chebdif(N,2)
        fs = 1./(2.*np.sqrt(np.pi * eps )) * np.exp(-(y-y0)**2 / 4./eps )
    else: 
        fs = kwargs['fs']
        N = fs.size
    
    I1 = np.identity(N); Z1 = np.zeros(N,dtype=np.complex)
    k2 = a**2 + b**2
    
    # Compute DeltaInv if not supplied
    if 'DeltaInv' not in kwargs:
        if ('D2' not in vars()): 
            DM = pseudo.chebdif(N,2)[1]
            D1 = DM[:,:,0]; D2 = DM[:,:,1]
        DeltaInv = np.linalg.solve( D2 - k2*I1, I1)
    else:
        DeltaInv = kwargs['DeltaInv']
    
    if 'D1' in kwargs: D1 = kwargs['D1']
    if 'D1' not in vars():
        DM = pseudo.chebdif(N,2)[1]
        D1 = DM[:,:,0]


    # Now to make Fs and FsAdj 
    Fs = np.zeros((4, 2*N),dtype=np.complex); FsAdj = Fs.copy()
    # Need Clenshaw-Curtis weights to define the adjoint of Fs
    weights = pseudo.clencurt(N)
    W2 = np.diag( np.concatenate((weights,weights)) )

    # Impulse along 'x'
    Fs[0,:N] = -1.j*a*DeltaInv @ (D1 @ fs) 
    Fs[0, N:]= 1.j*b*fs
    FsAdj[0] = (1./k2) *  np.concatenate(( -1.j*a*D1@fs, -1.j*b*fs  ))

    # Impulse along 'y'
    Fs[1,:N] =  -(a**2+b**2)*DeltaInv @ fs 
    # Doesn't contribute to forcing omega_y
    FsAdj[1] = np.concatenate(( fs, Z1 ))

    # Impulse along 'z'
    Fs[2,:N] = -1.j*b*DeltaInv @ (D1 @ fs) 
    Fs[2,N:] = -1.j*a*fs
    FsAdj[2] = (1./k2) *  np.concatenate(( -1.j*b*D1@fs,  1.j*a*fs  ))

    for k in range(3):
        FsAdj[k] = W2 @ FsAdj[k]

    Fs[3] = Fs[0] + Fs[1] + Fs[2]
    FsAdj[3] = FsAdj[0] + FsAdj[1] + FsAdj[2]


    return {'Fs':Fs, 'FsAdj':FsAdj, 'a':a, 'b':b,  'DeltaInv':DeltaInv}



def timeMap(a,b,tArr=None, linInst=None,modeDict=None, eddy=False, impulseArgs=None,printOut=False):
    """ Returns the Fourier coefficients (and energy) for u,v,w at a specified Fourier mode and time.
    Inputs:
        a, b    :      Streamwise and spanwise wavenumbers
        tArr    :      Array containing times at which responses are requested
        linInst(=None): Instance of ops.linearize
        modeDict(=None):Dictionary containing 'N' (number of wall-normal nodes), Re, and either 'turb'=True/False 
        eddy (=False):  If True, use eddy viscosity
        impulseArgs with entries (by priority):
            Fs (=None): Forcing function, 3C vector (see sec. 10.1 in Jovanovic's thesis) of size 3N
            Fsadj(=None): Adjoint of the forcing function
                    both of them need to be supplied. If either of them s not supplied,
            fs (=None): Forcing profile for each direction, of size N 
            y0, eps : Define the shape of the impulse (along any direction as)
                fs = 1./(2.*np.sqrt(np.pi * eps )) * np.exp(-(y-y0)**2 / 4./eps )

    Outputs:
        outDict with keys
            a, b, tArr, N, Re, turb (bool), eddy, impulseDict: all of the input arguments, and
            coeffArr: Fourier coefficients at time t
            energyArr : Energy in u, v, w at time t
    """
    if linInst is None:
        # Ensure modeDict has entries for N and Re. turb defaults to False 
        assert set(( 'N','Re' )) <= set(modeDict)
        modeDict['turb'] = modeDict.get('turb', False)
        linInst = ops.linearize(flowClass='channel', **modeDict)
    N = linInst.N
    
    # Construct system matrices for the dynamical system:
    #   d_t psi  = A  psi + Fs  delta(t)
    #   [u, v, w]= C  psi ,         where psi = [v, omega_y] 

    systemDict = linInst.makeSystem(a=a, b=b, adjoint=True, eddy=eddy)
    A = systemDict['A']; Aadj = systemDict['Aadj']
    C = systemDict['C']; Cadj = systemDict['Cadj']
    DeltaInv = systemDict['DeltaInv']
    # DeltaInv is the inverse of the matrix (D2- (a**2+b**2)I)

    # Build Impulse forcing vector Fs in the evolution equation if not supplied
    # FsAdj is also needed, for calculating the energy. 
    # Ideally, I can compute the energy directly from the velocity mode, but
    #   because of the messed up convention of using the weird Q matrix, I need this
    if impulseArgs is None: 
        warn("impulseArgs not supplied. Using impulse with y0=-0.9,eps=1/2000 ...")
        impulseArgs = {'N':N, 'y0':-0.9, 'eps':1./2000.}
        impulseDict = impulseVec(a,b,**impulseArgs)
    elif ('Fs' not in impulseArgs) or ('FsAdj' not in impulseArgs):
        impulseArgs['N'] = N
        assert ( set(('N', 'y0','eps')) <= set(impulseArgs) ) or ( 'fs' in impulseArgs ),\
                "Need either fs or (N,y0,eps) to build impulse vectors Fs, FsAdj"
        # I have DeltaInv and D1 from systemDict and linInst, so..
        impulseArgs.pop('DeltaInv',None); impulseArgs.pop('D1',None)
        impulseDict = impulseVec(a,b,DeltaInv=DeltaInv,D1=linInst.D1, **impulseArgs )
    else: 
        impulseDict = {'Fs':impulseArgs['Fs'], 'FsAdj':impulseArgs['FsAdj'], 'a':a, 'b':b, 'DeltaInv':DeltaInv}
    Fs = impulseDict['Fs']
    FsAdj = impulseDict['FsAdj']
    assert Fs.shape[1] == 2*N

    if tArr is None: tArr = np.array([1.])
    if printOut:
        print("Computing impulse response for times ",tArr)

    # At each time 't', and for 4 impulse cases (x, y, z, xyz), 3 components and 3 energies
    coeffArr = np.zeros( (tArr.size, 4, 3*N), dtype=np.complex )
    energyArr = np.zeros( (tArr.size,4, 4), dtype=np.complex)
    
    for i0 in range(tArr.size):
        t = tArr[i0]
        expFactor = C @ expm(t*A)
        expFactorAdj = expm(t*Aadj) @ Cadj
        for i1 in range(4):
            FsVec = Fs[i1].reshape((2*N,1))
            FsAdjVec = FsAdj[i1].reshape((1,2*N))
            coeffs = ( expFactor @ FsVec ).flatten()
            energy = (FsAdjVec @ expFactorAdj ).flatten() * coeffs  # (Weighted) pointwise energy in mode for u, v, w
            coeffArr[i0,i1] = coeffs 
            energyArr[i0,i1,0] = np.sum( energy[   : N ]   )
            energyArr[i0,i1,1] = np.sum( energy[ N :2*N]   )
            energyArr[i0,i1,2] = np.sum( energy[2*N:   ]   )
            energyArr[i0,i1,3] = np.sum( energy    )
    
    outDict = {'a':a, 'b':b, 'tArr':tArr, \
            'N':linInst.N, 'Re':linInst.Re, 'turb':linInst.flowState, \
            'eddy':eddy, 'impulseDict':impulseDict, 'coeffArr':coeffArr, 'energyArr':energyArr}
    return  outDict




def H2norm(aArr, bArr, Re=2000., N=41, turb=False, eddy=False, 
        epsArr = np.array([1./2000.]), y0Arr= np.array([-0.9]),
        tau=None):
    """ Compute H2 norms for Fourier modes
    Inputs:
        aArr :  Range of streamwise wavenumbers
        bArr :  Range of spanwise wavenumbers
        Re (= 2000): Reynolds number
        N  (= 41 ) : Number of collocation nodes (internal)
        turb (=False):  If True, use turbulent base flow, otherwise laminar
        eddy (=False):  If True, use eddy viscosity in the dynamics matrix
        tau (=None): If set to a positive integer, 
    Outputs:
        H2normArr   of shape (aArr.size,bArr.size)
    """
    linInst = ops.linearize(N=N, Re=Re, turb=turb, flowClass='channel')
    H2normArr = np.zeros((aArr.size, bArr.size, epsArr.size, y0Arr.size, 4))
    I2 = np.identity(2*N, dtype=np.complex)
    if (tau is not None) and (tau > 0.):
        print("Using exponential discounting for finite time horizon, over tau=", tau)
        diffA = -(1./tau)* I2
    else: diffA = 0.*I2

    fsArr = np.zeros((epsArr.size, y0Arr.size, N)) # This doesn't change much
    y = linInst.y
    for i2 in range(epsArr.size):
        eps = epsArr[i2]
        for i3 in range(y0Arr.size):
            y0 = y0Arr[i3]
            fsArr[i2,i3] = 1./(2.*np.sqrt(np.pi * eps )) * np.exp(-(y-y0)**2 / 4./eps )
    print("Running for aArr=",aArr)
    for i0 in range(aArr.size):
        a = aArr[i0]
        print("a=",a)
        for i1 in range(bArr.size):
            b = bArr[i1]

            systemDict = linInst.makeSystem(a=a,b=b,eddy=eddy, adjoint=True)
            A = systemDict['A']; Aadj = systemDict['Aadj']
            C = systemDict['C']; Cadj = systemDict['Cadj']
            DeltaInv = systemDict['DeltaInv']
            
            # In case finite time horizon is to be used
            A = A + diffA
            Aadj = Aadj + diffA

            Y = solve_sylvester( Aadj, A, -Cadj @ C )
            for i2 in range(epsArr.size):
                eps = epsArr[i2]
                for i3 in range(y0Arr.size):
                    y0 = y0Arr[i3]
                    impulseFunDict = impulseVec(a,b,fs=fsArr[i2,i3], DeltaInv=DeltaInv, D1=linInst.D1)
                    for i4 in range(4):
                        Fs = impulseFunDict['Fs'][i4].reshape((2*N,1))
                        FsAdj = impulseFunDict['FsAdj'][i4].reshape((1,2*N))

                        H2normArr[i0,i1,i2,i3,i4] = np.real(np.trace( FsAdj @ Y @ Fs ))

    return H2normArr


