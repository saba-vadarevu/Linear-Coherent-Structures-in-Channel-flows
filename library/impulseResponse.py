# impulseResponse.py
""" 
Define functions that take as parameters the Fourier wavenumbers, time, direction and amplitude of impulse, 
and return the Fourier coefficients or energy at that particular time.
Also, H2norms for inifinite time and finite time horizons.
"""
import numpy as np
from numpy.linalg import matrix_power
import ops
import pseudo
from warnings import warn
from scipy.linalg import expm, solve_sylvester
from scipy.sparse.linalg import expm_multiply
import pdb
from miscUtil import _arrGCD, _floatGCD, _areSame, _nearestEntry, _nearestInd

def _fs0(**kwargs):
    if kwargs.get('turb', True):
        # Turbulent case
        if 'Re' not in kwargs: Re = 2000.; warn("Re not supplied. Using 2000...")
        else : Re = kwargs['Re']
        if 'y0' not in kwargs: 
            y0plus =200.;
            y0 = -1. + y0plus/Re
            warn("y0 not supplied. Using %.3g so that y0^+=200..."%y0)
        else : y0 = kwargs['y0']
        if 'eps' not in kwargs: eps = 50.; warn("eps not supplied. Using 50...")
        else : eps = kwargs['eps']
        if 'N' not in kwargs: N = 251; warn("N not supplied. Using 251...")
        else : N = kwargs['N']

        y, DM = pseudo.chebdif(N,2)
        fs0 = 1./(2.*np.sqrt(np.pi * eps )) * np.exp(- (Re*(y-y0))**2 / 4./eps )
    else :
        # For laminar case, use the impulse of Jovanovic's thesis
        if 'Re' not in kwargs: Re = 2000.; warn("Re not supplied. Using 2000...")
        else : Re = kwargs['Re']
        if 'y0' not in kwargs: 
            y0 = -0.9 
            warn("y0 not supplied. Using %.3g..."%y0)
        else : y0 = kwargs['y0']
        if 'eps' not in kwargs: eps = 1/2000.; warn("eps not supplied. Using %.3g..."%eps)
        else : eps = kwargs['eps']
        if 'N' not in kwargs: N = 41; warn("N not supplied. Using %d..."%N)
        else : N = kwargs['N']
        y, DM = pseudo.chebdif(N,2)
        fs0 = 1./(2.*np.sqrt(np.pi * eps )) * np.exp(-(y-y0)**2 / 4./eps )
    fsDict = {'N':N, 'Re':Re, 'y0':y0, 'eps':eps, 'fs0':fs0, 'y':y, 'DM':DM}
    fsDict.update(kwargs)
    return fsDict

def _fs(fsAmp=None, **kwargs):
    fsDict = _fs0(**kwargs)
    fs0 = fsDict['fs0']
    if fsAmp is None : fsAmp = np.array([1.,0.,0.]) # [1*fs0, 0*fs0, 0*fs0]
    fsAmp = fsAmp.reshape(( fsAmp.size//3, 3))
    fs = np.zeros((fsAmp.shape[0], 3*fs0.size))

    w = pseudo.clencurt(fs0.size); w3 = np.concatenate((w,w,w))
    normalize = lambda arr : arr/ np.sqrt( w3 @ (np.abs(arr)**2 ))

    for m in range(fsAmp.shape[0]):
        amps = fsAmp[m]
        fs[m] = np.concatenate(( amps[0]*fs0, amps[1]*fs0, amps[2]*fs0 ))
        fs[m] = normalize(fs[m])
    fsDict.update({'fsAmp':fsAmp, 'fs':fs})
    return fsDict




def timeMap(a,b,tArr,fsAmp=None, coeffs=False, linInst=None, modeDict=None, eddy=False, impulseArgs=None,printOut=False):
    """
    Return energy and Fourier coeffs (if requested) for a single Fourier mode for different combinations of forcing at different times We're using the same shape for the impulse in y for each of x,y,z, but changing the relative amplitudes of fx, fy, fz. 
    Inputs:
        Compulsory args:
            a, b, tArr: wavenumbers and time
        keyword args:
            fsAmp (=None):      Relative amplitudes of fx, fy, fz, of shape (m,3)
                                    If not supplied, use [[1,0,0]]
                                    The total energy in [fx,fy,fz] will be normalized
            coeffs (=False):    If True, return Fourier coeffs as well
            linInst (=None):    Instance of class linearize(). 
            modeDict (=None):   If linInst is not supplied, use modeDict to build linInst
                                    keys:   'N', 'Re', 'turb'
            eddy (=False):      If True, use eddy viscosity
            impulseArgs (=None):Build impulse function (wall-normal using this)
                                    keys:   'fs' 
                                    If 'fs' is not supplied, use
                                        'y0', 'eps', linInst.N, linInst.Re to build fs using _fs()
    Outputs:
            outDict: dict with keys 
                        'energyArr', 'coeffArr' (if input kwargs 'coeffs' is True)
                        'fs', 'y0', 'eps', 'a', 'b', 'tArr', 'N', 'Re', 'turb', 'eddy','fsAmp'
                        If 'fs' was supplied as input to function, set y0 and eps to None
    """

    #=====================================================================
    # Figure out if tArr is uniformly/linearly spaced or not 
    #===========================

    # If tArr is uniformly spaced, then e^{A.n.dt} = e^{A.dt} @ e^{A.(n-1).dt} can be used
    # If this isn't the case, but tArr is still integral multiples of some dt, then 
    #       at least e^{A.n.dt} = (e^{A.dt})^n can be used
    # If neither is true, just do e^{A.t} = expm(A*t)
    tArr = np.array([tArr]).flatten()
    t0 =np.min(tArr) ;
    assert (tArr >= 0.).all()
    if _areSame(tArr, t0*np.arange(tArr.size) ): 
        uniformTime = True; linearTime = True
        tGCD = t0
    else :
        tGCD = _arrGCD(tArr)
        # If I have to raise e^{A.dt} to very large powers, it's not worth it
        if tGCD//t0 <= 5:
            uniformTime = False ; linearTime = True 
        # so, if tGCD is significantly smaller than t0, treat tArr to not be linear and use expm() all the time
        else : uniformTime = False; linearTime = False
    # These aren't used until the final loop where expFactor is calculated

    #=====================================================================
    # Decide if forcings are handled directly or a 3d basis is used
    #=========================
    if fsAmp is None:
        fsAmp = np.array([1,1,1]).reshape((1,3))
    else :
        fsAmp = np.array(fsAmp).reshape(( fsAmp.size//3, 3 ))

    # Because of linearity, I don't need to compute for different fs arrangements
    # Its sufficient to compute for fs =[fs0;0;0], [0;fs0;0], [0;0;fs0], 
    #   and then superpose the velocity fields 
    # Computing energy from superposed velocity fields is trivial
    if fsAmp.shape[0] > 3:
        useBasis = True
        fsAmp0 = np.array([1.,0.,0., 0.,1.,0., 0.,0.,1.]).reshape((3,3))
    else :
        useBasis = False
        fsAmp0 = fsAmp


    #====================================================================
    # System matrices
    #=======================
    if linInst is None :
        # Ensure modeDict has entries for N and Re. turb defaults to False 
        assert set(( 'N','Re' )) <= set(modeDict)
        modeDict['turb'] = modeDict.get('turb', False)
        linInst = ops.linearize(flowClass='channel', **modeDict)
    N = linInst.N
    
    # Construct system matrices for the dynamical system:
    #   d_t psi  = A  psi + B @ fs  delta(t)
    #   [u, v, w]= C  psi ,         where psi = [v, omega_y] 
    # or
    #   u = C_u psi;    v = C_v psi;    w = C_w psi
    # So that u(t) = C_u @ (e^{A(t/n)})^n @ B @ fs, and similarly for v, w 
    systemDict = linInst.makeSystem(a=a, b=b, adjoint=False, eddy=eddy)
    A = systemDict['A']
    C = systemDict['C']; 
    B = systemDict['B']; 
    W = linInst.weightDict['W1']   
    w = linInst.w
    
    #=================================================================
    # Forcing vectors 
    #==============
    # Function to build Fs for each fsAmp[k]
    if impulseArgs is None :
        impulseArgs = {'eddy':eddy, 'turb':(linInst.flowState=='turb')}
    else :
        impulseArgs.update({'eddy':eddy, 'turb':(linInst.flowState=='turb') })
    fsDict = _fs(fsAmp=fsAmp0, **impulseArgs)
    fs = fsDict['fs']; 
    fs = fs.T   # _fs returns fs so that fs[k] refers to a particular forcing, i.e. shape mx 3N
    # We want fs to a matrix of shape 3Nxm, so that B@fs is shape 2Nxm, and final u(t) is shape 3Nxm
    Fs = B @ fs

    if useBasis: 
        # To ensure I don't mess up the weighting factors, 
        # I want forcing to have magnitude unity, i.e. ||fs|| = 1
        # Let's denote such forcing with fs~
        # fs~ = fs/||fs|| = [Ax*fs0,Ay*fs0,Az*fs0]/||fs||, where ||fs|| = ||fs0|| sqrt{Ax^2+Ay^2+Az^2} = ||fs0|| |A|
        # Since fx~ = [fs0~;0;0] , fy~ = [0;fs0~;0] and fz~ = [0;0;fs0~], I can write fs~ as
        # fs~ = Ax/|A| fx~ + Ay/|A| fy~ + Az/|A| fz~
        # Ignore the abuse of notation for the amplitude A here..
        fsAmpNorm = np.sqrt(np.sum(fsAmp**2, axis=1)).reshape((fsAmp.shape[0],1))
        scaleFactors = fsAmp/fsAmpNorm  # scaleFactors[k] is [Ax/|A|, Ay/|A|, Az/|A|]
        scaleFactors = scaleFactors.T   # Make it of shape 3 x m, to allow
        #                               [ a1x; a2x; a3x; a4x;...]_3xm
        # us_3Nxm = [ux; uy; uz]_3Nx3   [ a1y; a2y; a3y; a4y;...]_3xm
        #                               [ a1z; a2z; a3z; a4z;...]_3xm
        # where a1x = Ax/|A| for the first forcing, a2x for the second forcing, and so on..
    


    #==================================================================
    # last bits of preparation
    #==================
    # Build energyArr by looping over fs.shape[0] and tArr
    energyArr = np.zeros((tArr.size,3, fsAmp.shape[0]))
    expFactor = np.identity(2*N, dtype=np.complex) 
    fs = fs.reshape((fs.size//(3*N), 3*N )).T   
    # So that fs is a matrix of shape 3N x m, allowing matrix multiplication u(t;fs) = C e^{At} B fs_{3N,m}
    if coeffs: coeffArr = np.zeros(( tArr.size, 3*N, fsAmp.shape[0]), dtype=np.complex)
    w1 = w.reshape((1,N))   # w1 is ClenCurt weight matrix, reshaped to a row vector
    def uvwEnergies(arr):
        # Use 2d arr, of shape 3Nx m
        returnArr = np.zeros((3, arr.shape[1]))
        returnArr[0:1] = w1@(np.abs(arr[:N])**2)
        returnArr[1:2] = w1@(np.abs(arr[N:2*N])**2) 
        returnArr[2:3] = w1@(np.abs(arr[2*N:])**2)
        return returnArr

    #==============================================================
    # And here we go...
    #======================
    if linearTime:
        expFactor0  = expm(A*t0)
    for i0 in range(tArr.size):
        if uniformTime:
            expFactor = expFactor @ expFactor0 
        elif linearTime:
            expFactor = matrix_power(expFactor0, int(round(tArr[i0]/tGCD)) )
        else :
            expFactor = expm(A * tArr[i0])

        tmpArr = C @ expFactor @ Fs     # This is a matrix of shape 3N x m 
        # Each column of tmpArr corresponds to field due to a forcing, fsAmp0[m]
        if not useBasis:
            # If I'm not computing for too many kinds of forcing, I don't have to split into Fx, Fy, and Fz and then superpose
            # For such cases, just calculate individually
            energyArr[i0] = uvwEnergies(tmpArr)
            if coeffs: coeffArr[i0] = tmpArr.copy()
        else :
            assert tmpArr.shape[1] == 3, " Need 3d basis for forcing along x,y,z"
            # When using too many forcings, first compute velocity fields individually for Fx, Fy, Fz
            # Now, compute velocity fields for each forcing given by fsAmp 
            # Need to be careful about how I weight each of these fields
            velVec = tmpArr @ scaleFactors 
            energyArr[i0] = uvwEnergies(velVec)
            if coeffs: coeffArr[i0] = velVec.copy()

    #===================================================================
    # Prepare dict to return...
    #===========================
    if impulseArgs is None : impulseArgs = {}
    impulseArgs.update(fsDict)
    impulseArgs.update({'fsAmp':fsAmp, 'N':N, 'Re':linInst.Re, 'a':a, 'b':b, 'tArr':tArr,\
            'turb':(linInst.flowState=='turb'), 'eddy':eddy })
    outDict = {'energyArr':energyArr}
    outDict.update(impulseArgs)
    if coeffs: outDict.update({'coeffArr':coeffArr})

    return outDict 



def evolveField(aArr, bArr, t, linInst=None,modeDict=None, eddy=False,
        fsAmp=None, impulseArgs=None):
    """
    Return Fourier coeffs a complete set of Fourier modes for a single forcing and time
    DO NOT CALL THIS DIRECTLY. Call from flowField.evolveImpulse() which returns a flowField instance.
    Inputs:
        Compulsory args:
            aArr, bArr, t: wavenumber sets and time
        keyword args:
            fsAmp (=None):      Relative amplitudes of fx, fy, fz, of size 3
                                    If not supplied, use [[1,0,0]]
                                    The total energy in [fx,fy,fz] will be normalized
            linInst (=None):    Instance of class linearize(). 
            modeDict (=None):   If linInst is not supplied, use modeDict to build linInst
                                    keys:   'N', 'Re', 'turb'
            eddy (=False):      If True, use eddy viscosity
            impulseArgs (=None):Build impulse function (wall-normal using this)
                                    keys:   'fs' 
                                    If 'fs' is not supplied, use
                                        'y0', 'eps', linInst.N, linInst.Re to build fs using _fs()
    Outputs:
            outDict: dict with keys 
                        'coeffArr'
                        'fs', 'y0', 'eps', 'a', 'b', 'tArr', 'N', 'Re', 'turb', 'eddy','fsAmp'
                        If 'fs' was supplied as input to function, set y0 and eps to None
    """
    #=====================================================================
    # Warn if aArr or bArr are not of the right form 
    #===========================
    a0 = np.min(np.abs(aArr[np.nonzero(aArr)])) 
    b0 = np.min(np.abs(bArr[np.nonzero(bArr)])) 
    aArr.sort(); bArr.sort()
    #pdb.set_trace()
    if not _areSame(aArr, a0*np.arange(-(aArr.size//2), aArr.size//2)):
        print("aArr isn't in ifft order... Have a look, aArr/a0 is:", aArr/a0)
    if not _areSame(bArr, b0*np.arange( bArr.size)):
        print("bArr isn't in ifft order... Have a look, bArr/b0 is:", bArr/b0)


    ################################
    if linInst is None :
        # Ensure modeDict has entries for N and Re. turb defaults to False 
        assert set(( 'N','Re' )) <= set(modeDict)
        modeDict['turb'] = modeDict.get('turb', False)
        linInst = ops.linearize(flowClass='channel', **modeDict)
    N = linInst.N
    W = linInst.weightDict['W1']   
    w = linInst.w
    
    #=================================================================
    # Forcing vector 
    #==============
    # Function to build Fs from fsAmp[k]
    if fsAmp is None : 
        warn("fsAmp not supplied, using [1,0,0]...")

    if impulseArgs is None :
        impulseArgs = {'eddy':eddy, 'turb':(linInst.flowState=='turb')}
    else :
        impulseArgs.update({'eddy':eddy, 'turb':(linInst.flowState=='turb') })
    if 'fsAmp' in impulseArgs: impulseArgs.pop('fsAmp')
    fsDict = _fs(fsAmp=fsAmp, **impulseArgs)
    fs = fsDict['fs']; 
    fs = fs.reshape((3*N,1))  

    #====================================================================
    # Define a function to handle mode-wise evolution to save space later
    #=======================
    # Construct system matrices for the dynamical system:
    #   d_t psi  = A  psi + B @ fs  delta(t)
    #   [u, v, w]= C  psi ,         where psi = [v, omega_y] 
    # So that [u v w] = C @ (e^{At}) @ Fs, and similarly for v, w 
    def modewiseEvolve(a,b):
        systemDict = linInst.makeSystem(a=a, b=b, adjoint=False, eddy=eddy)
        C = systemDict['C'] 
        B = systemDict['B']
        if t == 0. :
            return (C @ B @ fs).flatten()
        else :
            A = systemDict['A']
            return (C @ expm(A*t) @ B @ fs).flatten()


    
    #==================================================================
    # Define arrays and do the looping 
    #==================
    coeffArr = np.zeros((aArr.size, bArr.size, 3*N), dtype=np.complex)

    for i0 in range(aArr.size):
        a = aArr[i0]
        for i1 in range(bArr.size):
            b = bArr[i1]
            if (a==0) and (b==0): 
                continue

            coeffArr[i0,i1] = modewiseEvolve(a,b)

    if impulseArgs is None : impulseArgs = {}
    impulseArgs.update(fsDict)
    impulseArgs.update({'fsAmp':fsAmp, 'N':N, 'Re':linInst.Re, 'aArr':aArr, 'bArr':bArr, 't':t,\
            'turb':(linInst.flowState=='turb'), 'eddy':eddy })
    outDict= impulseArgs
    outDict.update({'coeffArr':coeffArr})

    return outDict 



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
    else : diffA = 0.*I2

    fsArr = np.zeros((epsArr.size, y0Arr.size, N)) # This doesn't change much
    y = linInst.y
    for i2 in range(epsArr.size):
        eps = epsArr[i2]
        for i3 in range(y0Arr.size):
            y0 = y0Arr[i3]
            fsArr[i2,i3] = 1./(2.*np.sqrt(np.pi * eps )) * np.exp(-(y-y0)**2 / 4./eps )
    #print("Running for aArr=",aArr)
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


