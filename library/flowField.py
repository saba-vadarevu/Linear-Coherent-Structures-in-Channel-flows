""" flowField.py 

Defines a class 'flowField' (inheriting numpy.ndarray) for plane channel and Couette flows
    Discretization is Fourier, Fourier, Chebyshev (collocation), Fourier in x,y,z
    z is spanwise
    Any operation that works for np.ndarray can be performed, but outputs will not be flowField instances
    Use class methods as much as possible.

Module functions outside the flowField class:
    getDefaultDict()
    impulseResponse:    Impulse response for specified modes, stored as flowField instance
    impulseResponse_split(): Impulse response run and stored in parts
    impulseResponse_add(): Improve earlier computations of impulse response by adding resolution (Fourier modes)
                            Needs testing
    checkFourierResolution: Check adequacy of resolution by comparing energy in the highest wavenumbers
    loadff: Load flowField instance from .mat file
    loadff_split(): Load flowField that was stored as parts (see impulseResponse_split())
"""

""" #####################################################
Sabarish Vadarevu
Melbourne School of Engineering
University of Melbourne, Australia 
May 2018
"""

import numpy as np
import scipy as sp
from scipy.io import loadmat, savemat
#from scipy.linalg import norm
from warnings import warn
import pseudo
import impulseResponse as impres
import ops
import pdb
from miscUtil import _areSame, _nearestEntry, _nearestInd
#from pseudo.py import chebint


defaultDict = {'Re': 2000.0, 'flowClass':'channel', 'flowState':'turb', 'eddy':False}
aArrDefault = np.linspace(0., 5., 64)
bArrDefault = np.linspace(0., 5., 128)
NDefault = 251

def getDefaultDict():
    return defaultDict.copy()

def impulseResponse_split(aArr, bArr, N,tArr, na,nb,**kwargs):
    """ 
    Call impulseResponse, but with aArr and bArr split into 'na' and 'nb' parts
    """
    if not (na==1): aStep = np.ceil(aArr.size/na)
    else : aStep = aArr.size
    aStep = int(aStep)
    if not (nb==1): bStep = np.ceil(bArr.size/ba)
    else : bStep = bArr.size
    bStep = int(bStep)
    aArrFull = aArr.copy()
    bArrFull = bArr.copy()

    if not ('fPrefix' in kwargs):
        raise RuntimeError("You have to supply fPrefix, or the fields aren't going to be saved")
    else :
        fPrefix0 = kwargs.pop('fPrefix')
    
    # In case some aParts have already been run,
    aInd0 = kwargs.pop('ignoreAParts', 0)
    for aInd in range(aInd0,na):
        aArr = aArrFull[ aInd*aStep : (aInd+1)*aStep ]
        if (not (na==1)):
            aPrefix = '_aPart%d_%d'%(aInd+1,na)
        else : 
            aPrefix = ''

        for bInd in range(nb):
            bArr = bArrFull[bInd*bStep : (bInd+1)*bStep]
            if (not (nb==1)):
                bPrefix = '_bPart%d_%d'%(bInd+1,nb)
            else :
                bPrefix = ''
            fPrefix = fPrefix0 + aPrefix + bPrefix 
            impulseResponse(aArr, bArr, N, tArr, fPrefix= fPrefix,**kwargs)

    return 

def impulseResponse_add(aArr, bArr, N, tArr, na, nb, loadPrefix, impulseArgs=None, **kwargs):
    """
        Pretty much like impulseResponse_split, except it uses existing ff and adds resolution
        Inputs:
            aArr, bArr: Fourier modes to be included in the final flowField
            N:  No. of wall-normal nodes
            tArr:   Time, should be t0*[0:m] or t0*[1:m] (choose only those where added resolution is needed)
            na: Number of parts to split aArr computation to. 
            nb: Same as na, but not allowed for now. Check back later.
            loadPrefix: Load existing ff files from this.
            kwargs accepts keys
                fsAmp : set to I_3x3 if not supplied
                impulseArgs: Dict with keys 'y0', 'eps', 'Re', 'N'

    """
    warn("impulseResponse_add() needs more testing. DO NOT TRUST RESULTS.")
    if nb != 1 :
        print("Not using nb. Only aArr is split into parts.")
        nb=1
    if impulseArgs is None :
        raise RuntimeError("impulseArgs must be supplied")
    assert set(('Re', 'y0', 'eps', 'N')) <= set(impulseArgs.keys())

    fsAmp = kwargs.get('fsAmp', np.identity(3))
    fsAmp = fsAmp.reshape((fsAmp.size//3, 3))
    t0 = np.amin(tArr[np.nonzero(tArr)]) 
    if not ('y0p' in loadPrefix):
        y0 = impulseArgs['y0']
        loadPrefix = loadPrefix + 'y0p%03d'%(round(-1000.*y0))

    # Use Fx field at t0 to figure out original resolution. 
    # And if any field has resolution lower than this, 
    #   suppose that it's be truncated by energy considerations, and ignore. 
    ff = loadff(loadPrefix+'_Fs_%d_%d_%d_t%05d.mat'%(
                    fsAmp[0,0], fsAmp[0,1], fsAmp[0,2], round(100.*round(t0,2)) ))
    flowDict = ff.flowDict.copy()
    aArr0 = ff.aArr.copy(); bArr0 = ff.bArr.copy()
    assert (aArr.size >= aArr0.size) and (bArr.size >= bArr0.size)
    aArrExistInd = np.in1d(aArr, aArr0); bArrExistInd = np.in1d(bArr, bArr0)     
    # Returns bool array saying if elements of aArr0 are in aArr
    aArrExtra = aArr[np.where(~aArrExistInd)]
    bArrExtra = bArr[np.where(~bArrExistInd)]
    assert (aArrExtra.size+aArr0.size == aArr.size) and (bArrExtra.size+bArr0.size == bArr.size)
        
    # I'll do the adding in 2 phases. First, for aArr0, compute for bArrExtra in a single step
    # That gives me flowField with aArr0 and bArr (i.e., bArr0+bArrExtra)
    # Then, for bArr, compute for aArrExtra and bArr in 'na' steps
    # Save the first phase ff as aPart0_na
    # And then the second phase ffs as aPartx_na
    # At the end, do append fields for 0 through na. 
    # Let impulseResponse do the computation and save parts. 
    # All I need to do is supply the right aArr, bArr, and fPrefix
    savePrefix = kwargs.get('savePrefix', loadPrefix)
    if not ('y0p' in savePrefix):
        y0 = impulseArgs['y0']
        savePrefix = savePrefix + 'y0p%03d'%(round(-1000.*y0))
    
    # Phase 1: aArr0, bArrExtra
    fPrefix = savePrefix + '_aPart0_%d'%na
    impulseResponse(aArr0, bArrExtra,ff.N, tArr, fsAmp=fsAmp, flowDict=flowDict, 
        impulseArgs=impulseArgs, fPrefix=fPrefix)

    # Phase 2: aArrExtra split into 'na' parts, bArr
    aStep = np.int(np.ceil(aArrExtra.size/na))
    for i0 in range(na):
        if i0 == (na-1):
            aArr1 = aArrExtra[ i0*aStep: ]
        else :
            aArr1 = aArrExtra[ i0*aStep: (i0+1)*aStep]
        fPrefix = savePrefix +'_aPart%d_%d'%(i0+1, na)
        impulseResponse(aArr1, bArr,ff.N, tArr, fsAmp=fsAmp, flowDict=flowDict, 
            impulseArgs=impulseArgs, fPrefix=fPrefix)

    # Now, all the files have been stored. Just gotta read them back and truncate
    forTimeSuffix = lambda fsVec, t: '_Fs_%d_%d_%d_t%05d.mat'%(
            fsVec[0], fsVec[1], fsVec[2], round(100.*round(t,2)) )
    for t in tArr:
        for i0 in range(fsAmp.shape[0]):
            fsVec = fsAmp[i0]
            fSuffix = forTimeSuffix(fsVec, t)
            fName = savePrefix + fSuffix
            try :
                ff = loadff(fName)
                for i1 in range(na+1):
                    fName  = savePrefix + '_aPart%d_%d'%(i1,na) + fSuffix
                    ff = ff.appendField(loadff(fName))
                    ff.sortWavenumbers()
            except :
                print("Could not do the appending for fsVec, t:", fsVec, t)
    return


def checkFourierResolution(ff, xTol=1.e-5, zTol=1.e-5, 
		    xModes=None, zModes=None, verbose=False):
    """
	Check if spectral flowfield is adequately resolved in Fourier modes
	Inputs:
	    ff: Spectral flowField 
	    xTol: %%!tolerance in x-modes
	    zTol: tolerance in z-modes
	Outputs:
	    None
    """
    L = ff.aArr.size//2 ; M = ff.bArr.size -1
    if verbose:
        print("Checking resolution adequacy for nx, nz = %d, %d"%(2*L, 2*M))
    normArr = ff.modeWiseNorm()
    normArrX = np.sum(normArr, axis=1)
    normArrX[1:L] = normArrX[1:L] + normArrX[-L+1:][::-1]
    normArrX = normArrX[:L+1]
    normArrZ = np.sum(normArr, axis=0)
    totalEnergy = np.sum(normArrX, axis=0)
    if xModes is None: xModes = L//5
    if zModes is None: zModes = M//5

    errFlag = False
    # Look at max energy in last L//5 modes, and ensure it's less than xTol of total
    for i1 in range(3):
        if ( np.amax(normArrX[-xModes:, i1]) >= xTol*totalEnergy[i1]):
            if verbose:
                print("u_%d is not adequately resolved in x"%i1)
                print("Ratio of max energy amongst last %d modes to total is %.2e while tol is %.2e"%(
                    xModes, np.amax(normArrX[-xModes:, i1])/totalEnergy[i1], xTol) )
            errFlag = True

        if ( np.amax(normArrZ[-zModes:, i1]) >= zTol*totalEnergy[i1]):
            if verbose:
                print("u_%d is not adequately resolved in z"%i1)
                print("Ratio of max energy amongst last %d modes to total is %.2e while tol is %.2e"%(
                    zModes, np.amax(normArrZ[-zModes:, i1])/totalEnergy[i1], zTol) )
            errFlag = True
    if not errFlag:
        print("Resolution is adequate to tolerances %.3g, %.3g"%(xTol, zTol))
    else :
        print("Resolution is NOT adequate to tolerances %.3g, %.3g"%(xTol, zTol))
    return (not errFlag)


def impulseResponse(aArr, bArr,N, tArr, fsAmp=None, flowDict=defaultDict, impulseArgs=None, fPrefix=None):
    """
    Generate flowField over set of Fourier modes as response to impulse
    Inputs:
        aArr:   Set of streamwise wavenumbers (>=0)
        bArr:   Set of spanwise wavenumbers (>=0). 
                    These are extended to cover the other 3 quadrants in a-b plane
                        when printing physical fields
        N:      Number of Cheb collocation nodes
        tArr:   Array of times to calculate response at
        flowDict (=defaultDict):
                Contains keys:
                Re (=2000):     Reynolds number
                flowClass (='channel'):  'channel'/'couette'/'bl' ('couette', 'bl' not supported)
                flowState (='turb')   :  'lam'/'turb'
                eddy (=False):  True/False
        impulseArgs (=None):
                see kwarg impulseArgs to impulseResponse.timeMap()
        fsAmp (=None): Forcing components [ax, ay, az] that multiplies fs0
                        For streamwise only, set fsAmp = np.array([1,0,0]).reshape((1,3))
        fPrefix (=None): If not None, save file with prefix fPrefix;
                            append forcing and time info to fPrefix
                        If None, do not save file. Return impresArray, collection of all flowfields, instead 
    Outputs:
        ffDict: dict containing flowField instances for each forcing in fsAmp 
                at t=tArr[0]. For other times, data must be saved
    """
    assert (bArr >= 0).all()
    tArr = np.array([tArr]).flatten()
    assert (tArr >= 0.).all()
    #====================================================
    # Warn if tArr isn't uniformly spaced, coz that gets too expensive
    #====================
    t0 = np.min(tArr[np.nonzero(tArr)])
    tArr0 = t0 * np.arange(tArr.size)
    if not (_areSame(tArr, tArr0) or _areSame(tArr, tArr0 + t0)):
        warn("tArr doesn't seem to be uniformly spaced. tArr/t0 is", tArr/t0)
    #===================================
    # Some misc stuff
    if (fPrefix is None):
        warn("No save name is specified. Will return the fields as a regular np.ndarray without saving to file...")

    aArr = aArr.flatten(); bArr = bArr.flatten()

    Re = flowDict['Re']; N = flowDict['N']
    if (impulseArgs is None) :
        warn("impulseArgs is not supplied. CHECK THE DEFAULTS FOR IMPULSE ARGS IN impres.timeMap().")
        impulseArgs = {'y0':-0.9}
    elif not (set(('y0','eps')) <= set(impulseArgs)) :
        warn("Not all required keys are supplied. CHECK THE DEFAULTS FOR IMPULSE ARGS IN impres.timeMap().")
        print("Supplied keys in impulseArgs are", impulseArgs.keys())
    impulseArgs.update({'Re':Re, 'N':N})

    #==============================
    # Create linInst if not supplied
    #=============
    if flowDict.get('flowState','turb') == 'turb':
        turb = True
    else : turb = False
    eddy = flowDict.get('eddy',False)
    print("Flow parameters are (Re,N,eddy,turb):",(Re,N,eddy,turb))
    linInst = ops.linearize(N=N, flowClass='channel',Re=Re,eddy=eddy,turb=turb)


    #=============================================================
    # Sort out fsAmp 
    #===============
    if fsAmp is None : 
        fsAmp = np.array([1.,0.,0.]).reshape((1,3))
    else :
        fsAmp = fsAmp.reshape((fsAmp.size//3,3))

    #================================================================
    # Start looping
    #======================
    # Create flowField instances for each t in tArr, one each for response to x, y, and z impulse
    # First, create one huge array to save them all
    impresArray = np.zeros((tArr.size, fsAmp.shape[0], aArr.size, bArr.size, 3, N),
            dtype=np.complex)
    # Save impulse responses into this array to start with, 
    #   and then loop over the first two axes to store flowField instances (parts)
    if (impulseArgs is not None) and ('y0' in impulseArgs) and ('eps' in impulseArgs):
        print("Running impres with y0p, eps : %.4g, %.4g" %(
                    Re*(1.+impulseArgs['y0']),impulseArgs['eps']) )
    print("And fsAmp:", fsAmp)
    print("For N and Re:", N, Re)
    for i0 in range(aArr.size):
        a = aArr[i0]
        print("a:",a)
        for i1 in range(bArr.size):
            b = bArr[i1]
            if (a==0.) and (b==0.): 
                continue            
            responseDict = impres.timeMap(a,b,tArr,linInst=linInst,
                    eddy=eddy,impulseArgs=impulseArgs, fsAmp=fsAmp, coeffs=True)
            coeffArr = responseDict['coeffArr']
            # This is of shape (tArr.size, 3N, fsAmp.shape[0])

            impresArray[:,:,i0,i1] = np.swapaxes(coeffArr, 1, 2).reshape(
                                    (tArr.size, fsAmp.shape[0], 3, N) )
    
    # If fPrefix is not supplied, return the numpy array
    if fPrefix is None :
        return impresArray
    #===============================================================================
    # Get flowField instances out of impresArray
    #=================
    # Base flowDict
    flowDict = {'Re':linInst.Re, 'flowState':linInst.flowState, 'flowClass':linInst.flowClass,\
            'eddy':eddy, 't':tArr[0]}    # Update 't' later on

    ff = flowField(aArr, bArr,N, flowDict=flowDict) 
    # Assign parts of impresArray to this ff and then save it using ff.saveff()
    for i0 in range(tArr.size):
        t = tArr[i0]
        ff.flowDict.update({'t':t})
        for i1 in range(fsAmp.shape[0]):
            fsVec = fsAmp[i1]
            # Normalize amplitudes in fsAmp so that the min is +/-1
            fsVec = fsVec/np.min(np.abs(fsVec[np.nonzero(fsVec)]))
            fName = fPrefix + '_Fs_%d_%d_%d_t%05d.mat'%(
                    fsVec[0], fsVec[1], fsVec[2], round(100.*t) )

            ff[:] = impresArray[i0,i1]

            ff.saveff(fName=fName, **impulseArgs)

    return impresArray 

def resolventMode(a,b,omegaArr,N,Re,
        turb=True,eddy=False,flowClass='channel',
        tArr=None,phaseArr=None,ampArr=None, modeNumber=1,
        modeSymm=2, symmLegend=['symm','anti','bottom','top']):
    """ Return flowField instance corresponding to (a,b)
        Inputs:
            (Positional)
            a : Streamwise wavenumber
            b : Spanwise wavenumber
            omegaArr: Frequencies; can be array, list, or float
            N : No. of cheb nodes
            Re: Reynolds number (ReTau for turbulent, ReCL for laminar)

            (Keyword)
            turb (=True)
            eddy (=False)
            flowClass (='channel')
            tArr (=None): Set of times (just 0 by default)
            phaseArr (=None): Phases for different omega-modes, 0 by default
            ampArr (=None): Amplitudes for different omega-modes, 1 by default
            modeNumber (=1): Can be float or np.ndarray (of size omegaArr.size) 
                            Allows probing resolvent modes that aren't the first
            symmLegend (=..): List containing strings that specify symmetries:
                            'bottom', 'top', 'symm', 'anti', meaning
                            predominatly in y<0, predominantly in y>0, symmetric, anti-symmetric
                            The actual symmetry imposed is given in modeSymm
            modeSymm (='bottom'): Symmetry of the resolvent mode (when two modes with equal singular values exist)
                            The symmetry is specified as symmLegend[modeSymm]
                            Can be int, or array of ints of shape (omegaArr.size,), (1,omegaArr.size), or (2, omegaArr.size);
                            first case applies same symm for all omega, second and third changes symm by omega but doesn't change for left and right leaning for a constant omega, fourth changes for each omega and each of left and right leaning modes

        Outputs:
            ff: flowField instance of shape (4,2,3,N)
                    4 streamwise wavenumbers are used: {0,a,-2a,-a}
                        to be consistent with the standard for flowField class
                    The coefficients for (-2a,b) and (0,b) are set to 0,
                        and calculated only for (a,b) and (-a,b)
    """
    #=========================================
    # Simple pre-processing
    #====================
    if (ampArr is not None) or (phaseArr is not None) or isinstance(modeNumber, (list,np.ndarray)) \
            or isinstance(modeSymm, (list,np.ndarray)):
        warn("Code needs more testing to run for more than a single Fourier mode at t=0")


    omegaArr = np.array([omegaArr]).flatten()
    if (a== 0.) or (b==0.): print("The code isn't meant for cases with either a or b being 0...")
    aArr = np.array([0.,abs(a),-2*abs(a),-abs(a)]) 
    bArr = np.array([0.,b])
   
    #"""
    warn("Calculating for (a,b,omega) and (-a,b,omega).")
    warn("To set (-a,b,omega) coefficient to zero, set ff[-1] = 0, ff is the returned flowField instance.")
    #"""
    if turb: flowState='turb'
    else : flowState='lam'
    
    #pdb.set_trace()
    # modeNumber prescribes the resolvent mode number to choose for each omega
    modeNumber = np.array([modeNumber]).flatten()
    if modeNumber.size==omegaArr.size:
        modeNumber = np.concatenate((modeNumber.flatten(),modeNumber.flatten())).reshape((2, omegaArr.size))
    elif modeNumber.size==2*omegaArr.size:
        modeNumber = modeNumber.reshape((2, omegaArr.size))
    else : 
        modeNumber = modeNumber[0]*np.ones((2,omegaArr.size),dtype=np.int)
    modeNumber[modeNumber<1] = 1
    modeNumber = modeNumber.astype(np.int)
    warn("Same number for resolvent mode used for left and right leaning waves")
    
    # modeSymm prescribes the wall-normall symmetry for each omega
    modeSymm = np.array([modeSymm]).flatten()
    if modeSymm.size==omegaArr.size:
        modeSymm = np.concatenate((modeSymm.flatten(),modeSymm.flatten())).reshape((2, omegaArr.size))
    elif modeSymm.size==2*omegaArr.size:
        modeSymm = modeSymm.reshape((2, omegaArr.size))
    else : 
        modeSymm = modeSymm[0]*np.ones((2,omegaArr.size),dtype=np.int)
    modeSymm = modeSymm%len(symmLegend)   
    modeSymm = modeSymm.astype(np.int)
    warn("Same wall-normal symmetry is used for left and right leaning waves")

    #========================================
    # Default phase: 0
    #===================
    if phaseArr is not None :
        if phaseArr.size == 2*omegaArr.size : 
            phaseArr = phaseArr.reshape((2,omegaArr.size))
        elif phaseArr.size == omegaArr.size:
            phaseArr = np.concatenate((phaseArr.flatten(),phaseArr.flatten())).reshape((2,omegaArr.size))
        else :
            warn("phaseArr.size is neither omegaArr.size nor 2*omegaArr.size")
            phaseArr = np.zeros((2,omegaArr.size))
    else : phaseArr = np.zeros((2,omegaArr.size))
    
    #========================================
    # Default amplitude: 1
    #===================
    if ampArr is not None :
        if ampArr.size == 2*omegaArr.size : 
            ampArr = ampArr.reshape((2,omegaArr.size))
        elif ampArr.size == omegaArr.size:
            ampArr = np.concatenate((ampArr.flatten(),ampArr.flatten())).reshape((2,omegaArr.size))
        else :
            warn("ampArr.size is neither omegaArr.size nor 2*omegaArr.size")
            ampArr = np.ones((2,omegaArr.size))
    else : ampArr = np.ones((2,omegaArr.size))

    #================================
    # Reflectional symmetry and assigning coeffs for 4 related Fourier modes:

    # Modes (a,b,omega), (a,-b,omega), (-a,b,-omega), and (-a,-b,-omega) are related;
    # The last two's coefficients should be complex conjugates of those of the first two
    #   for real-valuedness
    # The first and second are, for a,omega > 0, forward travelling waves that go     
    #   left and right for differing polarity of b
    # We ignore omega dependence, with the understanding that omega=a*c, and c is fixed 
    # flowField class assumes (a,b) and (-a,b) are both populated.
    # So, populate coeffs of (-a,b) from coeffs of (a,b) using reflectional symmetry,
    #   [u,v,w](x,y,-z) = [u,v,-w](x,y,z), 
    # which leads to the following relations between Fourier coefficients:
    # u_{a,b} = conj(u_{-a,b}) = u_{a,-b} = conj(u_{-a,-b})
    # The same relations hold for v, while for w, 
    # w_{a,b} =-conj(w_{-a,b}) =-w_{a,-b} = conj(w_{-a,-b})
    # So, for u and v, swapping the sign of 'b' doesn't change the coefficient, 
    #   while swapping the sign of 'a' makes it complex conjugate
    # For w, swapping the sign of 'b' makes the coefficient negative, 
    #   while swapping the sign of 'a' makes it complex conjugate and negative
    #===========
    if (omegaArr.size > 1) or (tArr.size > 1):
        warn("Code needs more testing to run for more than a single Fourier mode at t=0")

    #==========================================
    # Get resolvent modes for mode (+a, +b) for each omega; see later for (-a,+b)
    #==================
    linInst = ops.linearize(N=N,flowClass=flowClass,Re=Re,turb=turb)
    modeArr = np.zeros((aArr.size, bArr.size, omegaArr.size, 3*N),dtype=np.complex)
    #pdb.set_trace()
    if a > 0. : ix0 = 1; ix1 = 3
    else : ix0 = 3; ix1 = 1
    for i2 in range(omegaArr.size):
        omega = omegaArr[i2]
        resDict = linInst.getResolventModes(a, b, omega, 
                nSvals=modeNumber[0,i2]+1, eddy=eddy)
        svals = resDict['svals']
        # resDict['velocityModes'] should be shape (n_singModes, 3*N)
        
        # Check if singular values occur in pairs
        nMode = modeNumber[0,i2] 
        if (nMode%2 == 0):
            nMode1 = nMode-1; nMode2 = nMode
        else :
            nMode1 = nMode; nMode2 = nMode+1

        if (np.abs( svals[nMode1-1]-svals[nMode2-1])/svals[nMode-1] > 1.e-6 ):
            # Singular values are not equal; use mode number 'nMode'(-1)
            # Above, we used tolerance for relative error of 1.e-6
            velMode = resDict['velocityModes'][nMode-1]
        else :
            # There is a pair of singular values
            # Apply symmetries to get the appropriate linear combination
            velMode1 = resDict['velocityModes'][nMode1-1].reshape((3,N))
            velMode2 = resDict['velocityModes'][nMode2-1].reshape((3,N))
            modeDict = ops._modeSymms(velMode1,velMode2, N=N)
            if modeDict['err'] > 1.e-6:
                print("The symmetries don't seem to hold; err=%.3g"%modeDict['err'])
            velMode = modeDict[symmLegend[modeSymm[0,i2]]].flatten()
            # modeDict has keys 'symm','anti','bottom', and 'top'
            # symmLegend is a list containing a subset of the above keys
            # symmLegend[modeSymm[.]] specifies which key to use

        if b > 0 :
            # 'b' is positive in self.bArr; so it's sign doesn't have to be swapped
            modeArr[ix0,1,i2] = velMode.copy() # The a used to compute resolvent mode
            # Now, for -a; u and v are just conjugates, w is negative conjugate
            velMode[2*N:] *= -1.
            modeArr[ix1,1,i2] = np.conj(velMode)
        else :
            # 'b' in the computation was negative, but we're populating for positive 'b'  
            # No change for u and v, negative for w
            velMode[2*N:] = -1.* velMode[2*N:]

            modeArr[ix0,1,i2] = velMode.copy()
            # Now, for -a; u and v are just conjugates, w is negative conjugate
            velMode[2*N:] *= -1.
            modeArr[ix1,1,i2] = np.conj(velMode)
   
        #"""
        warn("For mode (-a,b), using reflexive symmetry about z=0 plane:")
        warn("[u,v,w](x,y,-z) = [u,v,-w](x,y,z)")
        warn("Applied as u_{a,b}=u_{a,-b}=conj(u_{-a,b})=conj(u_{-a,-b})")
        warn("This symmetry can be broken (shifted) by setting unequal phases or amplitudes for left and right leaning waves")
        #"""
    

    #=============================================
    # Exponent of phase, e^i*phi
    #===========
    expPhaseArrLong = np.ones((4,2,omegaArr.size,1),dtype=np.complex)   # Remember, aArr.size=4, bArr.size=2
    expPhaseArrLong[ 1,1,:,0] = np.exp(1.j*phaseArr[0])# (a,b) 
    expPhaseArrLong[-1,1,:,0] = np.exp(1.j*phaseArr[1])# (-a,b)
    #=========================
    # Amplitudes
    #===========
    ampArrLong = np.ones((4,2,omegaArr.size,1),dtype=np.complex)   # Remember, aArr.size=4, bArr.size=2
    ampArrLong[ 1,1,:,0] = ampArr[0]# (a,b) 
    ampArrLong[-1,1,:,0] = ampArr[1]# (-a,b)
    
    modeArr = modeArr*expPhaseArrLong*ampArrLong

    if tArr is None : tArr = np.array([0.])
    else : tArr = np.array([tArr]).flatten()

    if (tArr.size == 1) and (tArr[0] == 0.):
        flowDict ={'Re':Re,'N':N,'flowState':flowState,'flowClass':flowClass,\
                'eddy':eddy, 't':tArr[0]}
        ff = flowField(aArr,bArr,N,flowDict=flowDict)
        #ff[:] = np.sum(modeArr,axis=2).reshape((aArr.size, bArr.size,3,N))
        ff[:] = modeArr.reshape((aArr.size, bArr.size, 3,N))
        return ff
    else :
        ffList = []
        for i3 in range(tArr.size):
            t = tArr[i3]
            # We consider modes (a,b,omega), (-a,b,-omega), and their negatives
            #  Defining c = |omega|/|a|, and setting omega = a*c accounts for the sign of omega
            cArr = np.abs(omegaArr/a).reshape((1,1,omegaArr.size,1))
            acArr= self.aArr.reshape((self.aArr.size,1,1,1))*cArr
            warn("Phase speed, c, is assumed to be positive")

            expArr = np.exp(-1.j*acArr*t)
            flowDict ={'Re':Re,'N':N,'flowState':flowState,'flowClass':flowClass,\
                    'eddy':eddy, 't':t}
            ff = flowField(aArr,bArr,N,flowDict=flowDict) # Initialize with zeros 
            ff[:] = np.sum(modeArr*expArr,axis=2).reshape((aArr.size,bArr.size,3,N))
            # modeArr contains coeffs for t=0; to get coeffs at a later t, 
            #   multiply by e^{-i*a*c*t} for each travelling wave, 
            #   and add all of these travelling waves to get field at time t

            ffList.append(ff)
         
        return ffList








def loadff(fName,printOut=True):
    """ Load flowField instance from .mat file
        Inputs:
        (Compulsory)
            fName: .mat file name (.mat appended if not in file name)
        (optional)
            printOut (=True): Report load success if True
        
        Outputs:
            flowField instance
    """
    if not fName.endswith('.mat'):
        fNamePrefix = fName.split('.')[0]
        fName = fNamePrefix +'.mat'

    loadDict =  loadmat(fName)
    # The mat files could have impulseArgs. To account for those, I'll split loadDict like so:
    # First, all the arguments that should be strings
    flowDict = {'flowClass': str(loadDict.pop('flowClass')[0]),\
            'flowState': str(loadDict.pop('flowState')[0]), \
            'eddy': str(loadDict.pop('eddy')[0]) }
    # And the int
    N =  int(loadDict.pop('N'))
    # The remaining stuff should all be either floats or arrays
    # The main arrays are aArr, bArr, and ffArr. Get them out first
    aArr = loadDict.pop('aArr').flatten()
    bArr = loadDict.pop('bArr').flatten()
    ffArr = loadDict.pop('ffArr')
    if False :
        # I'm not going to be using the extra stuff. So don't bother with them
        # Everything else should be either float or arr.  
        for key in loadDict.keys():
            if isinstance(loadDict[key], np.ndarray):
                # Save size 1 np.ndarrays as floats
                #if loadDict[key].size == 1 : flowDict[key] = float(loadDict.pop(key))
                #else : flowDict[key] = loadDict.pop(key).flatten()
                if loadDict[key].size == 1 : flowDict[key] = float(loadDict[key])
                else : flowDict[key] = loadDict[key].flatten()
                # And others in flattened form
    else :
        # Except for Re and t
        flowDict['Re'] = float(loadDict.pop('Re'))
        flowDict['t'] = float(loadDict.pop('t'))
       
    # Finally: 
    ff = flowField(aArr, bArr, N, flowDict=flowDict)
    #pdb.set_trace()
    ff[:] = ffArr.reshape((aArr.size, bArr.size, 3, N))

    if printOut:
        print("Loaded flowField from ",fName)

    return ff


def loadff_split(fPrefix, t, fsAmp=None, na=32, nb=1, **kwargs):
    """
    Load flowField from a set of  .mat files 
    Current convention for naming split flowfield files is
        fPrefix + '_aPart3_32' + '_Fs_x_x_x' + '_txxxxx.mat'  # when only aArr is split but not bArr
        fPrefix + '_bPart5_16' + '_Fs_x_x_x' + '_txxxxx.mat'  # when only bArr is split
        fPrefix + '_aPart3_32' + '_bPart5_16'+ '_Fs_x_x_x' + '_txxxxx.mat'
        t is stored in 5 digits, with last 2 digits representing decimals
    Inputs:
        fPrefix: Prefix that identifies the case, such as 'ffEddyRe10000'
        t: time (float)
        fsAmp (=None): Forcing component vector [ax, ay,az]. 
                        Default is [1,0,0], representing Fx
        na (=32):   Number of parts for aArr
        nb (=1) :   Number of parts for bArr
    Outputs:
        flowField instance 
    """
    if fsAmp is None : fsAmp = np.array([1., 0., 0.])   # Default forcing: Fx
    fsAmp = np.array([fsAmp]).flatten() # In case it's supplied as a list
    fsAmp = fsAmp/np.min(np.abs(fsAmp[np.nonzero(fsAmp)]))  # Normalize fsAmp

    fSuffix = '_Fs_%d_%d_%d_t%05d.mat'%(
            fsAmp[0], fsAmp[1], fsAmp[2], round(100.*t) )
    aPrefix = ''; bPrefix = ''
    firstField = True
    for aInd in range(na):
        if na > 1:
            aPrefix = '_aPart%d_%d'%(aInd+1, na)
        for bInd in range(nb):
            if nb > 1:
                bPrefix = '_bPart%d_%d'%(bInd+1,nb)

            fName = fPrefix + aPrefix + bPrefix + fSuffix
            
            # Don't want to append the first field to itself, so..
            if not firstField:
                ff = ff.appendField( loadff(fName,**kwargs) )
            else :
                ff = loadff(fName,**kwargs)
                firstField = False 
    ff.sortWavenumbers()
    return ff
    


class flowField(np.ndarray):
    """
    Defines a class (inheriting numpy.ndarray) for plane channel and Couette flows
        Discretization is Fourier, Fourier, Chebyshev (collocation), Fourier in x,y,z
        z is spanwise
    "self" refers to an instance of the class

    Class instances can have shape (l1, m1, 3, N) 
        where l1 = aArr.size, m1 = bArr.size, N is number of Cheb nodes 
            Cheb nodes should be internal only, but it is possible to rewrite the class to allow walls.
            See kwargs in the functions in pseudo.py
        aArr and bArr can have arbitrary wavenumbers, but it's advised to have 
            aArr = a0*{0,1,..,L-1,-L,-L+1,..,-1}, bArr = b0*{0,1,...,M}
            so fft-like methods work
        Class attributes are:
            aArr, bArr, N, y, flowDict, D1, D2, D, weightDict, U, dU, d2U
                aArr, bArr, and N are as above
                y is array of Chebyshev nodes
                flowDict is a dict containing keys
                    Re :Reynolds number based on normalization of U (U_CL for laminar, U_tau for turbulent, usually)
                    flowClass: 'channel', 'couette', or 'bl'; 'bl' is not fully supported
                    flowState: 'lam' or 'turb' (used to initialize U profile if not supplied)
                    eddy: True/False    
                    t: float, time normalized by U and h used for normalizing LNSE
                D1, D: Differentiation operator for first wall-normal derivative
                D2 : Second derivative
                weightDict: Matrices that help with Clenshaw-Curtis quadrature. Dict with keys
                    w:  arr of shape (N,) for quadrature
                    W1:  diagonal matrix of shape (N,N) for quadrature
                    W2, W3: As W1, but for 2-c and 3-c vectors instead of scalars
                    W1Sqrt, W2Sqrt, W3Sqrt: Square roots
                    W1SqrtInv, W2SqrtInv, W3SqrtInv: Inverse of square-roots
                    These are useful to define weighted matrices in SVD
                U, dU, d2U: Mean velocity and its derivatives

    Class methods are:
        saveff: Save to .mat file , 
        verify: simple checks on flowDict and self.shape
        copyArray: copy to a np.ndarray without the flowField machinery
        ddx, ddy, ddz, ddx2, ddy2, ddz2 :  Derivatives in x, y, z; z is spanwise
        laplacian, curl  
            divergence is easy to compute: self.ddx()[:,:,0]+self.ddy()[:,:,1]+self.ddz()[:,:,2]; excluded since it's a scalar and not 3-c vector)
        swirl: Field of imaginary components of complex eigvals; see Zhou et. al. (1999) JFM, Mechanisms of...
        dot, norm: integrate over x,y,z and divide by box volume
        toPhysical: physical field from the spectral field in self
        savePhysical: Higher level than toPhysical; save a common set of fields to .mat file 
        appendField: Append coefficients from two sets of wavenumbers to produce a larger field for the union of the wavenumbers. Restrictions apply
        sortWavenumbers: To be used with appendField to get modes into fft-order
        zero, identity: self.identity() =self, self.zero() has shape and attributes of self, but zero elements.
        slice: Interpolate/extrapolate 
        modeWiseNorm: Integral along wall-normal for each Fourier mdoe

    Initialization:
        flowField() creates an instance using a default dictionary: 
            a 3 component zero-vector of shape (64,128,3,251) for turbulent channel flow at Re=2000 (see top of module for defaultDict)
        flowField(aArr, bArr,N,flowDict=dictName) creates an instance with shape (aArr.size, bArr.size,3,N), with flowDict to specify the flow conditions.
    A warning message is printed when the default dictionary is used.
    """
    def __new__(cls, *args, flowDict=None,**kwargs):
        """Creates a new instance of flowField class, call as
            flowField(aArr, bArr, N ,flowDict={'N', 'Re', 'flowClass','flowState','eddy','t'})
        """
        if flowDict is None :
            flowDict= defaultDict
        else :
            assert set(('Re','flowClass', 'flowState','eddy','t')) <= set(flowDict)
        if len(args) > 0 : aArr = args[0]
        else : aArr = aArrDefault 
        if len(args) > 1 : bArr = args[1]
        else : bArr = bArrDefault 
        if len(args) > 2 : N = args[2]
        else : N = NDefault 
       
         
        aArrPos = aArr[ np.where(aArr>=0.)[0]]
        aArrNeg = aArr[ np.where(aArr<0.)[0]]
        aArr = np.concatenate(( np.sort(aArrPos), np.sort(aArrNeg)    ))
        # Need aArr to go as a0*[ 0, 1, 2, ..., L-1, L, -L+1, -L+2,...,-1]
        
        bArr = np.sort(bArr)
        arrShape =  (aArr.size,bArr.size,3,N)
        obj = np.ndarray.__new__(cls,
                shape=arrShape, dtype=np.complex, buffer=np.zeros(arrShape, dtype=np.complex) )
                
        # Attributes:
        # flowDict, aArr, bArr, N, y, D1, D2, D, weightDict, U, dU, d2U
        obj.flowDict = flowDict.copy()
        obj.aArr = aArr
        obj.bArr = bArr
        obj.N = N
        
        warn("Using channel geometry, excluding walls; see kwargs of pseudo.chebdif() and pseudo.weightMats() for differentiation/integration matrices that could include walls.")
        y,DM = pseudo.chebdif(N,2)
        D1 = np.ascontiguousarray(DM[:,:,0]) 
        D2 = np.ascontiguousarray(DM[:,:,1])
        obj.y = y; obj.D1 = D1; obj.D2 = D2; obj.D = D1
       
        obj.weightDict = pseudo.weightMats(N)

        assert flowDict['flowClass'] not in ['couette','bl'], "Currently only channel is supported, Couette/BL are easily implemented with a few fixes"
        if flowDict['flowState'] == 'lam':
            U = 1. - y**2; dU = -2.*y; d2U = -2.*np.ones(N)
        else :
            turbDict = ops.turbMeanChannel(N=N,Re=flowDict['Re'])
            U = turbDict['U']; dU = turbDict['dU']; d2U = turbDict['d2U']

        obj.U = U; obj.dU = dU; obj.d2U = d2U

        return obj
        
    
    def __array_finalize__(self,obj):
        """ For copying/slicing flowField instances"""
        if obj is None : return
         
        self.flowDict = getattr(self,'flowDict',obj.flowDict.copy())
        self.aArr = getattr(self,'aArr',obj.aArr)
        self.bArr = getattr(self,'bArr',obj.bArr)
        self.N = getattr(self,'N',obj.N)
        self.y = getattr(self,'y',obj.y)
        self.D = getattr(self,'D',obj.D)
        self.D1 = getattr(self,'D1',obj.D1)
        self.D2 = getattr(self,'D2',obj.D2)
        self.U = getattr(self,'U',obj.U)
        self.dU = getattr(self,'dU',obj.dU)
        self.d2U = getattr(self,'d2U',obj.d2U)
        self.weightDict = getattr(self,'weightDict',obj.weightDict)
        return

    def saveff(self, fName, **kwargs):
        """ Save flowField instance to .mat file
        Inputs:
            Compulsory args:
                fName:  File name (.mat appended if format unspecified)
            kwargs:
                Anything is accepted. kwargs will be directly saved to the .mat file
                Particularly useful to pass impulseArgs, the parameters used to define the impulse that produced this field
        Outputs:
            None """
        if not fName.endswith('.mat'):
            fNamePrefix = fName.split('.')[0]
            fName = fNamePrefix +'.mat'
        saveDict = {'aArr':self.aArr, 'bArr':self.bArr, 'N':self.N, 'U':self.U, 'ffArr':self.copyArray()}
        saveDict.update(self.flowDict)
        saveDict.update(**kwargs)

        savemat(fName, saveDict)
        print("Saved flowField data to ",fName)

        return

    
    def verify(self):
        """Ensures that the size of the class array is consistent with dictionary entries. 
        Use this when writing new methods or tests"""
        assert set(( 'Re','flowClass', 'flowState', 'eddy' )) <= set(self.flowDict)
        assert self.flowDict['flowClass'] == 'channel'
        assert self.shape == (self.aArr.size, self.bArr.size, 3, self.N )
        return
    
    
    
    def copyArray(self):
        """ Returns a copy of the np.ndarray of the instance. 
        This is useful for manipulating the entries of a flowField without bothering with all the checks"""
        return self.view(np.ndarray).copy()
    
    
    def ddx(self):
        """ Returns a flowField instance that gives the partial derivative along "x" """
        partialX = 1.j * self.aArr.reshape((self.aArr.size,1,1,1)) * self
        return partialX
    
    def ddx2(self):
        """ Returns a flowField instance that gives the second partial derivative along "x" """
        partialX2 = -1.* (self.aArr**2).reshape((self.aArr.size,1,1,1)) * self
        return partialX2
    
    def ddz(self):
        """ Returns a flowField instance that gives the partial derivative along "z" (spanwise) """
        partialZ = 1.j * self.bArr.reshape((1,self.bArr.size,1,1)) * self
        return partialZ
    
    def ddz2(self):
        """ Returns a flowField instance that gives the second partial derivative along "z" (spanwise)  """
        partialZ2 = -1.* (self.bArr**2).reshape((1,self.bArr.size,1,1)) * self
        return partialZ2
    
    
    def ddy(self):
        """ Returns a flowField instance that gives the partial derivative along "y" """
        N = self.N
        partialY = self.copy()
        tempArr = self.reshape(self.size//N,N)
        partialY[:] = ( tempArr @ self.D1.T ).reshape(partialY.shape)
        return partialY
    
    def ddy2(self):
        """ Returns a flowField instance that gives the second partial derivative along "y" """
        N = self.N
        partialY2 = self.copy()
        tempArr = self.reshape(self.size//N,N)
        partialY2[:] = ( tempArr @ self.D2.T ).reshape(partialY2.shape)
        return partialY2
    
        
    def laplacian(self):
        """ Computes Laplacian for a flowField instance """
        return self.ddx2() + self.ddy2() + self.ddz2()
            
    
    def curl(self):
        """ Computes curl of vector field as [w_y-v_z, u_z - w_x, v_x - u_y]"""
        curlInst = self.zero()
        # Curl: [ w_y-v_z, u_z-w_x, v_x-u_y ]
        tmpArr = self.ddx()
        curlInst[:,:,1] += -tmpArr[:,:,2]      # [0, -w_x, 0]
        curlInst[:,:,2] +=  tmpArr[:,:,1]      # [0, -w_x, v_x]
        tmpArr = self.ddy()
        curlInst[:,:,0] +=  tmpArr[:,:,2]      # [w_y, -w_x, v_x]
        curlInst[:,:,2] += -tmpArr[:,:,0]      # [w_y, -w_x, v_x-u_y]
        tmpArr = self.ddz()
        curlInst[:,:,0] += -tmpArr[:,:,1]      # [w_y-v_z, -w_x, v_x-u_y]
        curlInst[:,:,1] +=  tmpArr[:,:,0]      # [w_y-v-z, u_z-w_x, v_x-u_y]

        return curlInst
   

    def dot(self, vec2):
        """Computes inner product for two flowField objects, scalar or vector,
            by integrating {self[kx,kz,kc]*vec2[].conj()} for each x_j, and adding the integrals for j=1,..,self.nd.
        Currently, only inner products of objects with identical dictionaries are supported"""
        assert (self.shape == vec2.shape), 'Method for inner products is currently unable to handle instances with different flowDicts'
        weightArr = self.weightDict['w'].reshape((1,1,1,self.N))
        dotProd = 0.5* np.sum( (self.conj() * vec2 * weightArr).flatten() )
        warn("Dot product is only defined for a>=0 and b>=0. The other quadrants are currently not included")
        warn("Spacing between Fourier modes is not accounted for in the dot product...")
        return dotProd
   

    def norm(self):
        """Integrates v[nd=j]*v[nd=j].conjugate() along x_j, sums across j=1,..,self.nd , and takes its square-root"""
        return np.sqrt(self.dot(self))




    def toPhysical(self, arr=None, x0=None, lx=None, z0=None, N=None, ySpace='cheb', doSort=False, ifft=True, **kwargs):
        """
        Get physical fields from spectral
        Inputs:
            arr:    Any 3-d or 4-d array of spectral coefficients for a scalar of shape consistent with self
            keyword arguments:
            x0, lx, z0; all None by default
                If they're not specified, use domain x in [0,Lx], z in [-Lz/2,Lz/2]
                If any of them is specified, truncate domain to [x0,x0+lx] , [z0,-z0]
                Allow x0 to go in [-Lx/2., 0] if needed, coz sometimes the full structure has some back-propogation
            N (=None):  If set to a value other than self.N, interpolate to a different wall-normal grid
            ySpace (='cheb'): If 'cheb', keep data on Chebyshev grid
                                If 'linear', interpolate to uniformly spaced points
            doSort (=True):  If True, call self.sortWavenumbers()
            ifft (=True): Use ifft or simple exponentiation to compute physical field
            **kwargs: accept keys 'L', 'M', 'nx', and 'nz' to use for padding
        Outputs:
            outDict with keyds
                arrPhys:    Physical field for arr
                xArr, yArr, zArr:   arrays for x, y, and z
        """
        if arr is None :
            self.sortWavenumbers()
            arr = self.copyArray()[:,:,0]
            warn("arr is not supplied to 'toPhysical', using streamwise velocity...")
        else : 
            assert (arr.shape == (self.aArr.size, self.bArr.size,self.N)) or \
                (arr.shape == (self.aArr.size, self.bArr.size,1,self.N))
            if ifft:
                warn("Ensure input arr confirms to fft order for wavenumbers")
        if doSort:
            self.sortWavenumbers()

        # fundamental wavenumbers to define periodic domain
        a0 = np.amin( self.aArr[ np.where(self.aArr > 0.)[0]] )   # Smallest positive wavenumber
        b0 = np.amin( self.bArr[ np.where(self.bArr > 0.)[0]] )   # Smallest positive wavenumber
        Lx = 2.*np.pi/a0; Lz = 2.*np.pi/b0
        
        if 'nx' in kwargs: nx = kwargs.pop('nx')
        elif 'L' in kwargs: nx = 2*kwargs.pop('L')
        else : nx = self.aArr.size
        nx = np.int(nx - nx%2)

        if 'nz' in kwargs: nz = kwargs.pop('nz')
        elif 'M' in kwargs: nz = 2*kwargs.pop('M')
        else : nz = 2*(self.bArr.size-1)
        nz = np.int(nz - nz%2)
        aArr = self.aArr; bArr = self.bArr
        if ifft :  
            # Use this when aArr.size and bArr.size are large
            if (aArr.size < 20) or (bArr.size < 10):
                warn("aArr.size and bArr.size are %d, %d. ifft is unlikely to produce a 'good' physical field. Set ifft=False in toPhysical()."%(aArr.size,bArr.size))
            #===============================================================
            # Ensure fft order for self.aArr and self.bArr 
            #================================
            # x-Modes go 0,1,..,L-1,L,-L+1,-L+2,..,-1, a total of 2*L
            # z-Modes go 0,1,..,M-1,M, a total of M+1
            L0 = self.aArr.size//2; M0 = self.bArr.size-1 
            L = nx//2;  M = nz//2


            #pdb.set_trace()

            # Ensure aArr and bArr are integral multiples
            aArrIdeal = a0 * np.fft.ifftshift( np.arange(-L0, L0) )   # -L is included, but L isn't. 
            bArrIdeal = b0 * np.arange(0,M0+1)
            if not ( np.linalg.norm(self.aArr- aArrIdeal) < 1.e-09*a0 ) :
                print("aArr doesn't seem to be integral multiples. Have a look")
                print("a0 is", a0)
                print("aArr/a0 is ", self.aArr/a0)
            if not ( np.linalg.norm(self.bArr- bArrIdeal) < 1.e-09*b0 ) :
                print("bArr doesn't seem to be integral multiples. Have a look")
                print("b0 is", b0)
                print("bArr/b0 is ", self.bArr/b0)

            #=================================================================
            # Define basic xArr, yArr, and zArr
            #=============================
            # Grids in x, z, and y
            # Worry about x0, x1, z0, z1 after the iFFT
            # Note that I have a different L and L0 if I'm trying to use padding
            # For the checks above, use L0, M0 since arr and self are defined for L0, M0
            # L and M come into play in _spec2physIfft(), so I must build xArr and zArr to reflect these
            # Of course, if L==L0 and M==M0, there's nothing to worry about
            xArr = np.linspace(0., Lx, nx+1)[:-1]
            zArr = np.linspace(-Lz/2., Lz/2., nz+1)[:-1]

            if L== L0 : Lifft = None
            else : Lifft = L
            if M== M0 : Mifft = None
            else : Mifft = M
            #==================
            # Get physical field
            #====
            arrPhysUnfolded = _spec2physIfft( arr, L=Lifft, M=Mifft)

        else :
            # Use this when aArr.size and bArr.size are small
            if (self.aArr.size > 40) and (self.bArr.size > 20):
                warn("aArr.size and bArr.size are %d, %d. Use ifft to improve speed by setting ifft=True in toPhysical()"%(self.aArr.size,self.bArr.size))
            #==================
            # Get physical field
            #======

            physDict = _spec2physManual(arr, self.aArr, self.bArr,
                    nx = nx, nz = nz)
            xArr = physDict['xArr']; zArr = physDict['zArr']
            arrPhysUnfolded = physDict['arrPhys']

        if (x0 is None ) and (lx is None) and (z0 is None) \
                and (ySpace == 'cheb'):
            #print("Returning without the index truncation")
            return {'arrPhys':arrPhysUnfolded,'xArr':xArr.flatten(), 'zArr':zArr.flatten(), 'yArr':self.y.copy()}
        
        
        #=================================================================
        # z0ind and z1ind
        #===================
       
        # Let's start working on truncating the domain according to x0,x1,z0
        if (z0 is None) or (z0>0.) or (z0 < -Lz/2.): z0 = zArr[0]
        z1 = -z0 - ( zArr[1] - zArr[0] )*(1.+1.e-4)    # +Lz/2. is excluded, so...
        try :
            z0ind = np.where(zArr <= z0)[0][-1]     # Index of largest entry of zArr <= z0 
        except :
            z0ind = 0   # Just in case..
        try :
            z1ind = np.where(zArr >= z1)[0][0] + 1  
            # Index of (second) smallest entry of zArr >= z0
        except :
            z1ind = zArr.size - 1
        

        # Slice in  z  
        arrPhysUnfolded = arrPhysUnfolded[:, z0ind:z1ind]
        zArr = zArr[z0ind:z1ind]
        nz1 = zArr.size

        #==================================
        # Get x0ind:
        #===============
        foldInX = False     
        # This flag tells me if I need to rearrange the x-dimension, when x0 < 0
        if (x0 is None): x0 = xArr[0]
        # The below code assumes -Lx/2<x0<Lx, x0<x1<Lx
        # Cases where reference frame moves beyond these limits can be reduced to the domain like so
        if x0 < 0.:
            # This is the tricky bit
            try :
                x0ind = np.where(xArr <= -x0)[0][-1]     
            except :
                x0ind = 0   # If np.where() doesn't work for whatever reason
                print("setting x0ind to 0")
            x0ind = max(x0ind,1)    # Ensure x0ind is at least 1 to avoid issues later
            # Say x0 is -0.3. 
            # I'll count the number of entries to +0.3 instead of -0.3, 
            #   which are the same because the grid is uniform.
            # Then, I'll get so many indices from the end (near Lx) and move them to the start
            foldInX = True
            nxFrame = 0     # This isn't useful for x0 in [-Lx/2, Lx]
            # For x0 > Lx, nxFrame tells me how many x-periods we have moved in
        else :
            nxFrame = x0 // Lx  
            x0 = x0 % Lx
            x0ind = np.where(xArr <= x0)[0][-1]     
        
        #===================================
        # Get x1ind:
        #=================

        # The starting point is set now. Next, x1
        if lx is None : lx = xArr[-1]
        lx = min(lx, xArr[-1])    # Don't keep a domain greater than Lx (for now)

        foldInX_x1 = False  # Same as foldInX above, but the other way around now
        x1 = x0 + lx
        x1 = max( 0, x1 )   # Don't allow x1 < 0 for now (makes the rearranging easier)
        if x1 > xArr[-1]:
            foldInX_x1 = True   
            # foldInX and foldInX_x1 can't be true at the same time, because lx < Lx
            x1 = x1 % Lx

        try :
            x1ind = np.where(xArr >= x1)[0][0] + 1   # Index of smallest entry of xArr >= x1
        except :
            x1ind = xArr.size - 1 
            print("setting x1ind to -1")
            # Sometimes x1 lies between xArr[-1] and Lx.
            # A bit of effort should let me handle this properly, but I can't be bothered

       
        # I should be able to do the following with just 2 cases instead of 3, 
        #   too lazy for that now... 
        if not (foldInX or foldInX_x1) : 
            # 0 < x0 < x1 < Lx
            indList = np.r_[x0ind:x1ind]
        elif foldInX:
            # x0 < 0 < x1 < Lx   
            indList = np.r_[-x0ind:0, 0:x1ind]
            xArr[np.r_[-x0ind:0]] += -Lx
        else :
            # 0 < x1 < x0 < Lx
            # Think of this as 0 < x0 < Lx < Lx+x1
            # Since [Lx, Lx+x1] is identical to [0, x1], the 0 < x1 < x0 < Lx condition's valid
            indList = np.r_[ x0ind:xArr.size, 0:x1ind ]
            xArr[:x1ind] += Lx

        xArr = xArr[indList] +  (Lx * nxFrame)
        arrPhys = arrPhysUnfolded[indList]
        nx1 = xArr.size


        #==============================================================
        # Divide by size of array if ifft was used, interpolate in y if needed
        #======================
        if ifft : arrPhys *= (1./(2.*np.pi)**2) * (a0*b0)

        interpFlag = True
        if N is not None :
            if ySpace == 'linear':
                yArr = np.linspace(1., -1., N+2)[1:-1]
            else : yArr = pseudo.chebdif(N,1)[0]
        else :
            if ySpace == 'linear':
                yArr = np.linspace(1., -1., self.N+2)[1:-1]
            else :
                yArr = self.y.copy()
                interpFlag = False

        if interpFlag:
            for i0 in range(nx1):
                for i1 in range(nz1):
                    arrPhys[i0,i1] = pseudo.chebint(arr[i0,i1], yArr)

        return {'arrPhys':arrPhys, 'xArr':xArr, 'zArr':zArr, 'yArr':yArr}




    def sortWavenumbers(self):
        """ 
            Sort wavenumbers to fft order (0,1,..,L-1, -L, -L+1,..,-1)
            Methods such as ddx(), div(), and curl() don't need the wavenumbers to be sorted
            But the ordering is important when calling any fft-related functions. 
            So, when doing appendField, append away without paying attention to the ordering.
            When they're all done, start sorting.

        """
        # if np.amax(self.aArr) > -np.amin(self.aArr):
        if False :
            # This is probably an old case where +L*a0 was kept instead of -L*a0
            # change things up a bit
            # CAN'T DO THIS ANYMORE, coz using this on incomplete fields (while appending) screws things up
            aMaxInd = np.argmax(self.aArr)
            aMax = self.aArr[aMaxInd]
            self.aArr[aMaxInd] = -aMax
            self[aMaxInd] = 0.
            print("Have set a=%.3g to %.3g"%(aMax, -aMax))
        aArr = self.aArr.copy().flatten()
        bArr = self.bArr.copy().flatten()
        a0 = np.amin(aArr[ np.where(aArr > 0.)[0]])   # Smallest positive wavenumber
        b0 = np.amin(bArr[ np.where(bArr > 0.)[0]])   # Smallest positive wavenumber
        assert (b0 >= 0.).all()
        
        L = aArr.size//2 ; M = bArr.size-1
        aArrIdeal = a0*np.arange(-L,L)
        bArrIdeal = b0*np.arange(0,M+1)

        if  (np.linalg.norm( aArr - np.fft.ifftshift(aArrIdeal) ) < a0* 1.e-09)  and \
                (np.linalg.norm( bArr - bArrIdeal ) < b0* 1.e-09):
            #Nothing to do here. Wavenumbers already in fft order
            return

            
        
        # Do the sorting in 2 steps.
        # First, sort in ascending order, and use numpy's ifftshift to get to fft order
        aInd = np.argsort(aArr)
        bInd = np.argsort(bArr)
        
        #pdb.set_trace()

        # Now, aArr[aInd] should look like aArrIdeal, and bArr[bInd] as bArrIdeal
        sortedFlag = True
        if  not (np.linalg.norm(aArr[aInd] - aArrIdeal) < 1.e-09*a0) :
            print("aArr's elements aren't integral multiples of a0; aArrSorted/a0:", aArr[aInd]/a0)
            sortedFlag = False
        if  not (np.linalg.norm(bArr[bInd] - bArrIdeal) < 1.e-09*b0) :
            print("bArr's elements aren't integral multiples of b0; bArrSorted/b0:", bArr[bInd]/b0 )
            sortedFlag = False

        # Now, aArr[aInd] and bArr[bInd] should be properly sorted.
        # Get them into fft order
        self.aArr = aArr[np.fft.ifftshift(aInd)]
        self.bArr = bArr[bInd]  # bArr goes 0 to M only
        
        self[:] = self[:,bInd]
        self[:] = self[np.fft.ifftshift(aInd)]
        if sortedFlag :
            print("Successfully sorted self, aArr, and bArr into fft order")

        return



    def swirl(self, doSort=True, **kwargs):
        """ Returns the swirling strength for the field in physical space
        IMPORTANT: aArr and bArr must be (positive) integral multiples of aArr[0] and bArr[0]
        This function runs only on numpy's ifft; the custom ifft is now dropped
        Inputs:             
            **kwargs; all of them are passed to flowField.toPhysical(). See its docstring
        Outputs:
            swirlDict with keys 
                swirl, xArr, yArr, zArr
        """
        if doSort: self.sortWavenumbers()
        else : 
            if kwargs.get('ifft',True):
                warn("We're now assuming wavenumbers are in fft order...")
        
        tmpArr = self.ddx()
        ux = tmpArr[:,:,0]
        vx = tmpArr[:,:,1]
        wx = tmpArr[:,:,2]

        uxDict = self.toPhysical(arr= ux, **kwargs)
        xArr = uxDict['xArr']; yArr = uxDict['yArr']; zArr = uxDict['zArr']
        uxPhys = uxDict['arrPhys']
        # The shapes of velGrad and swirlStrength depend on parameters  x0,x1,z0,z1 in **kwargs
        velGrad = np.zeros( (uxPhys.shape[0], uxPhys.shape[1], uxPhys.shape[2], 3,3) )
        velGrad[:,:,:,0,0] = uxPhys 
        velGrad[:,:,:,1,0] = self.toPhysical(arr= vx, **kwargs)['arrPhys']
        velGrad[:,:,:,2,0] = self.toPhysical(arr= wx, **kwargs)['arrPhys']        
        ux = None; vx = None; wx = None # Just in case ux and others aren't just pointers 

        tmpArr = self.ddy()
        uy = tmpArr[:,:,0]
        vy = tmpArr[:,:,1]
        wy = tmpArr[:,:,2]
        velGrad[:,:,:,0,1] = self.toPhysical(arr= uy, **kwargs)['arrPhys']
        velGrad[:,:,:,1,1] = self.toPhysical(arr= vy, **kwargs)['arrPhys']
        velGrad[:,:,:,2,1] = self.toPhysical(arr= wy, **kwargs)['arrPhys']
        uy = None; vy = None; wy = None

        tmpArr = self.ddz()
        uz = tmpArr[:,:,0]
        vz = tmpArr[:,:,1]
        wz = tmpArr[:,:,2]
        velGrad[:,:,:,0,2] = self.toPhysical(arr= uz, **kwargs)['arrPhys']
        velGrad[:,:,:,1,2] = self.toPhysical(arr= vz, **kwargs)['arrPhys']
        velGrad[:,:,:,2,2] = self.toPhysical(arr= wz, **kwargs)['arrPhys']
        uz = None; vz = None; wz = None
        tmpArr = None
        
        swirlStrength = _velGrad2swirl(velGrad)
        
        return {'swirl':swirlStrength, 'xArr':xArr, 'yArr':yArr, 'zArr':zArr} 

    def savePhysical(self, fieldList=['u'], fName=None,fPrefix=None,fsAmp=None, **kwargs):
        """ 
        Save physical fields to .mat files
        Inputs:
            fieldList:  List of strings corresponding to the fields to be saved. Acceptable strings are:
                'u', 'v', 'w' for velocity components
                'vorx', 'vory', 'vorz' for vorticity components
                'swirl' for swirl
                'div' for divergence
            fName (=None): File name to save to
            **kwargs:   these are sent to flowField.toPhysical(); refer to its docstring
        Outputs:
            None (just saving)
        """
        self.sortWavenumbers()
        if fName is None : 
            if fPrefix is None : 
                fName = 'testPhysFields.mat'
            else :
                if fsAmp is None : 
                    fsAmp = np.array([1., 0., 0.])
                    warn("fsAmp not supplied to savePhysical; assuming [1,0,0]")
                else :
                    fsAmp = np.array([fsAmp]).flatten()
                    fsAmp = fsAmp/np.amin( np.abs(fsAmp[np.nonzero(fsAmp)]) )
                fName = fPrefix + '_Fs_%d_%d_%d_t%05d.mat'%(
                        fsAmp[0],fsAmp[1],fsAmp[2], round(100*self.flowDict['t']))

        if not fName.endswith('.mat'): fName = fName.split('.')[0] + '.mat'

        saveDict = {}
        savedList = []

        if 'u' in fieldList: 
            physDict =  self.toPhysical(arr=self[:,:,0], **kwargs)
            saveDict.update({'u':physDict['arrPhys']} )
            savedList.append('u')
        if 'v' in fieldList: 
            physDict =  self.toPhysical(arr=self[:,:,1], **kwargs)
            saveDict.update({'v':physDict['arrPhys']} )
            savedList.append('v')
        if 'w' in fieldList: 
            physDict =  self.toPhysical(arr=self[:,:,2], **kwargs)
            saveDict.update({'w':physDict['arrPhys']} )
            savedList.append('w')
        if 'swirl' in fieldList: 
            physDict =  self.swirl(**kwargs)
            saveDict.update({'swirl':physDict['swirl']} )
            savedList.append('swirl')
        if ('vorx' in fieldList) or ('vory' in fieldList) or ('vorz' in fieldList):
            vorticity = self.curl()
            if 'vorx' in fieldList:
                physDict =  self.toPhysical(arr=vorticity[:,:,0], **kwargs)
                saveDict.update({'vorx':physDict['arrPhys']} )
                savedList.append('vorx')
            if 'vory' in fieldList:
                physDict =  self.toPhysical(arr=vorticity[:,:,1], **kwargs)
                saveDict.update({'vory':physDict['arrPhys']} )
                savedList.append('vory')
            if 'vorz' in fieldList:
                physDict =  self.toPhysical(arr=vorticity[:,:,2], **kwargs)
                saveDict.update({'vorz':physDict['arrPhys']} )
                savedList.append('vorz')
        
        a0 = np.amin( np.abs( self.aArr[ np.nonzero(self.aArr)] ))
        b0 = np.amin( np.abs( self.bArr[ np.nonzero(self.bArr)] ))
        Lx = 2.*np.pi/a0; Lz = 2.*np.pi/b0
        turbInt = (2./Lx/Lz)* np.sum( np.sum( self.conj() * self, axis=1), axis = 0 ) 
        uvInt = (2./Lx/Lz)* np.sum( np.sum( self[:,:,0].conj() * self[:,:,1], axis=1), axis = 0 ).reshape((1,self.N))
        turbInt = np.concatenate(( turbInt, uvInt), axis=0)

        saveDict.update({'turbInt':turbInt})
        

        if len(saveDict.keys()) == 0:
            warn("Looks like there were no acceptable strings in fieldList:"+str(fieldList))
        else :
            saveDict.update({'xArr':physDict['xArr'], 'yArr':physDict['yArr'], 'zArr':physDict['zArr']})
            saveDict.update(self.flowDict)
            savemat(fName, saveDict)
            print("Saved fields   %s   to file %s"%(str(savedList), fName) )
        
        return

    def streakHeight(self, threshold=0.25, lowerHalf=True, bisect=False, **kwargs):
        """ Height of streamwise velocity streak, defined as distance from wall of isosurface
            Inputs:
                (keyword)
                threshold (=0.25): Fraction of uMax that defines the isosurface
                lowerHalf (=True): If True, compute height for y in [-1,0], 
                                    if False, compute for y in [0,1]
                bisect (=False)  : If True, interpolate and use bisection to improve accuracy
                kwargs: Passed to self.toPhysical()
        """
        uDict = self.toPhysical(arr=self[:,:,0], **kwargs)
        uPhys = uDict['arrPhys']
        N = self.N
        
        # Max over x and z, considering absolute values
        uMax_xz = np.amax( np.amax( np.abs(uPhys), axis=1), axis=0)
        if lowerHalf :
            uMax_xz = uMax_xz[N//2:][::-1] # y goes from 1 to -1; now considering [-1,0]
            yTmp = self.y[N//2:][::-1]; yWall = -1.
            # Now, uMax_xz and yTmp go from wall to core
        else :
            uMax_xz = uMax_xz[:N//2]
            yTmp = self.y[:N//2]; yWall = 1.
            # uMax_xz and yTmp go from wall to core already

        if bisect:
            warn("Bisection isn't presently implemented. Do it later if needed")
        
        # Normalize uMax_xz
        uMax_xz = uMax_xz/np.amax(uMax_xz)
        # Define streak height as smallest (or largest if lowerHalf=False) y where 
        #    uMax_xz > threshold
        if (threshold >= 1.) or (threshold <= 0.):
            threshold = 0.25
            warn("threshold should be in (0,1); resetting to 0.25")

        yInd = np.amax( np.where( uMax_xz > threshold )[0].flatten() )
        # Largest index where (normalized) uMax_xz is above the threshold

        strHeight = np.abs( yWall - yTmp[yInd] )
        return strHeight


    def ReynoldsStresses(self):
        """ Return a dict containing Reynolds stresses (averaged in streamwise-spanwise)"""
        a0 = np.amin( np.abs( self.aArr[ np.nonzero(self.aArr)] ))
        b0 = np.amin( np.abs( self.bArr[ np.nonzero(self.bArr)] ))
        Lx = 2.*np.pi/a0; Lz = 2.*np.pi/b0
        turbInt = np.real((2./Lx/Lz)* np.sum( np.sum( self.conj() * self, axis=1), axis = 0 ) )
        uvInt = np.real((2./Lx/Lz)* np.sum( np.sum( self[:,:,0].conj() * self[:,:,1], axis=1), axis = 0 ))
        uwInt = np.real((2./Lx/Lz)* np.sum( np.sum( self[:,:,0].conj() * self[:,:,2], axis=1), axis = 0 ))
        vwInt = np.real((2./Lx/Lz)* np.sum( np.sum( self[:,:,1].conj() * self[:,:,2], axis=1), axis = 0 ))

        stressDict = {'uu':turbInt[0], 'vv':turbInt[1], 'ww':turbInt[2], 'uv':uvInt, 'uw':uwInt, 'vw':vwInt}
        return stressDict

    def eddyInt(self,lowerHalf=True):
        """ Return eddy intensity functions (normalized Reynolds stresses)"""
        stressDict = self.ReynoldsStresses()
        if (self.aArr.size < 30) or (self.bArr.size < 15):
            strHeight = self.streakHeight(ifft=False, nx=50,nz=50,lowerHalf=lowerHalf)
        else :
            strHeight = self.streakHeight(ifft=True, lowerHalf=lowerHalf)

        # Scale y by strHeight
        N = self.N
        if lowerHalf:
            i0 = N-1; i1 = N//2; iStep=-1; yWall = -1.
        else :
            i0 = 0; i1 = N//2; iStep = 1; yWall = 1.
        i0 = 0; i1 = N; iStep=1
        yArr = np.abs(self.y[i0:i1:iStep]-yWall)/strHeight
        # Scale intensities by uv
        uv = stressDict['uv']; uvAbsMax = np.amax(np.abs(uv))
        uv = (uv/uvAbsMax)[i0:i1:iStep]
        uu = (stressDict['uu']/uvAbsMax)[i0:i1:iStep]
        vv = (stressDict['vv']/uvAbsMax)[i0:i1:iStep]
        ww = (stressDict['ww']/uvAbsMax)[i0:i1:iStep]
        uw = (stressDict['uw']/uvAbsMax)[i0:i1:iStep]
        vw = (stressDict['vw']/uvAbsMax)[i0:i1:iStep]
        
        #warn("eddyInt defined for half-channel, according to kwargs 'lowerHalf'(=True)")
        warn("Normalized height for eddyInt is returned as eddyIntDict['yArr']")
        
        eddyIntDict = {'uu':uu,'vv':vv,'ww':ww,'uv':uv,'uw':uw,'vw':vw,'yArr':yArr,'dh':strHeight}
        return eddyIntDict

    def zero(self):
        """Returns an object of the same class and shape as self, but with zeros as entries"""
        obj = self.copy()
        obj[:] = 0.
        return obj


    def identity(self):
        """ Returns itself (no copies are made)"""
        return self

    def appendField(self, ff):
        """ Combine fields from different flowField instances
        This isn't very complicated since modes evolve independently, 
            and the spec2phys routines don't use iFFT.
        IMPORTANT: ONLY WORKS IF EITHER aArr or bArr ARE IDENTICAL IN SELF AND FF
        For a more generalized version, use flowField.messyAppendField()

        Inputs:
            ff: flowField that needs to be appended
                    Must have all attributes to be identical to self except for 
                        either aArr xor bArr 
        Outputs:
            ffLong: appended flowField
        """
        # Ensure sorted a and b arrs so that the later stuff makes sense
        #self.aArr = np.sort(self.aArr); self.bArr = np.sort(self.bArr)
        #ff.aArr = np.sort(ff.aArr); ff.bArr = np.sort(ff.bArr)
        #aArrInd0 = np.argsort(self.aArr); bArrInd0 = np.argsort(self.bArr)
        #aArrInd1 = np.argsort(ff.aArr);   bArrInd1 = np.argsort(ff.bArr)

        assert all([self.flowDict[key] == ff.flowDict[key] for key in self.flowDict])
        assert self.N == ff.N 

        if (self.aArr.size == ff.aArr.size) and (self.aArr == ff.aArr).all():
            #assert not (self.bArr == ff.bArr).any()
            bNew = np.concatenate(( self.bArr.flatten(), ff.bArr.flatten() )) 
            ffLong = flowField( self.aArr, bNew, self.N, flowDict = self.flowDict )
            ffLong[:, :self.bArr.size] = self
            ffLong[:, self.bArr.size:] = ff
            #ffLong.sortWavenumbers() 
        
        elif (self.bArr.size == ff.bArr.size) and (self.bArr == ff.bArr).all():
            #assert not (self.aArr == ff.aArr).any()
            aNew = np.concatenate(( self.aArr.flatten(), ff.aArr.flatten() )) 
            ffLong = flowField( aNew, self.bArr, self.N, flowDict = self.flowDict )
            ffLong[:self.aArr.size] = self
            ffLong[self.aArr.size:] = ff
            #ffLong.sortWavenumbers()
        else :
            raise RuntimeError("flowfields must have matching aArr xor bArr")
        return ffLong
    
    def slice(self, L=None, M=None, N=None):
        """
        Interpolate/pad flowField along x, y, z
        Inputs:
            L (=None):  Number of streamwise Fourier modes. If more than self.nx//2, pad with zeros. If less, drop higher wavenumbers
            M (=None):  Same as above, for spanwise
            N (=None):  Number of (internal) wall-normal nodes
        Outputs:
            flowField instance of shape, (2L, 2M, 3, N)
        """
        self.sortWavenumbers()
        if (L is None) and (M is None) and (N is None):
            return self.copy()
        aArr = self.aArr.copy(); bArr = self.bArr.copy()
        a0 = aArr[1]; b0 = bArr[1]  # I've already done self.sortwavenumbers(), so..
        if L is None : L = self.aArr.size//2
        if M is None : M = self.bArr.size - 1 
        if N is None : N = self.N
        N = int(N)
        L0 = self.aArr.size // 2; M0 = self.bArr.size - 1; N0 = self.N
        aArrNew = np.fft.ifftshift( a0 * np.arange(-L, L) )
        bArrNew = np.arange(M+1)

        # Initialize a new zero ff instance
        ff = flowField(aArrNew, bArrNew, N0, flowDict=self.flowDict)
        # If N0 != N, first do the Fourier mode slicing, and do the interpolation later

        # Now to set the entries of ff from self
        # First, the a>0 modes 
        L = min(L, L0); M = min(M,M0)   # These modes exist in both self and ff
        ff[:L, :M+1] = self[:L, :M+1]
        # Now the a<0 modes
        ff[-L:, :M+1] = self[-L:, :M+1]
        # That takes care of padding/dropping modes. 

        # Now to the interpolation, if needed
        if (N == N0):
            return ff

        ffNew = flowField(aArrNew, bArrNew, N, flowDict=self.flowDict)
        
        for i0 in range(ff.aArr.size):
            for i1 in range(ff.bArr.size):
                for i2 in range(3):
                    ffNew[i0,i1,i2] = pseudo.chebint( ff[i0,i1,i2], ffNew.y )

        return ffNew

    def modeWiseNorm(self):
        """ 
        Return energy in each Fourier mode for each component of velocity
        Inputs:
            None
        Outputs:
            array flattened in last dimension (along wall-normal)
        """
        w = pseudo.clencurt(self.N).reshape((1,1,1,self.N))

        energyArr = np.sum( w * self.conj() * self, axis=-1 ).real
        return energyArr


def _spec2physIfft(arr0, L=None, M=None):
    """
       Inputs:
        (Compulsory)
            arr0: Array of shape (2*L, M+1, N) of spectral coefficients
            L (=None):      If supplied, and different from self.aArr//2, slice flowField
                                Used for padding, since numpy's padding seems a bit funny
            M (=None):      Same as above, but for spanwise modes
    """
    assert arr0.ndim == 3
    L0 = arr0.shape[0]//2; M0 = arr0.shape[1]-1; N= arr0.shape[2]
    warn("Assuming that modes go positive and negative in kx")
    if (L is not None) or (M is not None): 
        if L is None : L = L0
        if M is None : M = M0
        arr = np.zeros((2*L, M+1,N), dtype=np.complex)
        Lt = min(L, L0); Mt = min(M,M0)
        arr[:Lt, :Mt+1] = arr0[:Lt, :Mt+1]
        arr[-Lt:, :Mt+1] = arr0[-Lt:, :Mt+1]
    else :
        arr = arr0; L = L0; M = M0
        

    # The array is now ready to be used for numpy's rifft2
    nx = 2*L; nz = 2*M

    scaleFactor = nx * nz    # times something related to a0,b0??
    physField =  scaleFactor * np.fft.irfft2( arr, axes=(0,1) )  
    
    # This field goes from 0 to Lz in z. I want to to go from -Lz/2 to Lz/2(exclusive):
    physField = np.concatenate(  (physField[:, nz//2:], physField[:, :nz//2]), axis=1)
    return physField

def _spec2physManual(arr0, aArr, bArr, xArr=None, zArr=None,nx=None, nz=None):
    """
       Inputs:
        (Positional) 
        arr0: Array of shape (l1, m1, N)  of spectral coefficients 
        aArr: Array of streamwise wavenumbers of size l1
        bArr: Array of spanwise wavenumbers of size m1
        (Keyword)
        xArr (=None): Streamwise locations. Use a0 from aArr and nx if not supplied
        zArr (=NOne): Spanwise locations. Use b0 from bArr and nz if not supplied
        nx (=None): Size of xArr if xArr is not supplied. If not supplied, use aArr.size
        nz (=None): Size of zArr if zArr is not supplied. If not supplied, use 2*(bArr.size-1)
    """
    bArr = bArr[ bArr >= 0.]   # Real-valuedness for b<0
    if xArr is None : 
        a0 = np.amin(np.abs(  aArr[np.nonzero(aArr)] ))
        if (nx is None) or not isinstance(nx, (int, np.integer)) : 
            nx = aArr.size
        xArr = np.linspace(0., 2.*np.pi/a0, num=nx, endpoint=False)
    xArr = xArr.reshape((xArr.size, 1,1)) 
    aArrIdeal = a0*np.arange(-(aArr.size//2), aArr.size//2)
    if not _areSame(np.sort(aArr), aArrIdeal):
        warn("aArr in _spec2physManual is not of form a0*{-L,..,L-1}")

    if zArr is None : 
        b0 = np.amin(np.abs(  bArr[np.nonzero(bArr)] ))
        if (nz is None) or not isinstance(nz, (int, np.integer)) : 
            nz = 2*(bArr.size-1)
        zArr = np.linspace(-np.pi/b0, np.pi/b0, num=nz, endpoint=False)
    zArr = zArr.reshape((1,zArr.size,1))

    N = arr0.shape[-1]
    assert arr0.ndim == 3

    arrPhys = np.zeros((xArr.size, zArr.size, N))
    for i0 in range(aArr.size):
        a = aArr[i0]
        for i1 in range(bArr.size):
            b = bArr[i1]
            if (b==0.) and (a >= 0.) : continue
            # Get these from coeffs for  (-|a|,0), instead of (|a|,0)

            arrPhys += 2.*np.real( arr0[i0,i1]* np.exp(1.j*(a*xArr + b*zArr) ) ) 

    return {'arrPhys':arrPhys, 'xArr':xArr.flatten(), 'zArr':zArr.flatten()}


def _velGrad2swirl(velGrad):
    """ Get swirl field from a (physical) velocity gradient tensor field
    Inputs:
        velGrad: velocity gradient tensor field of shape (nx, nz, ny, 3, 3)
    Outputs:
        swirl: swirl field of shape (nx, nz, ny)
    """
    a = velGrad # Makes the code simpler to write
    assert velGrad.ndim == 5
    
    # Eigenvalues (s) of velGrad at each point are solutions to the cubic
    # s^3 + ps + q = 0, where 
    #       p = -0.5*trace(velGrad @ velGrad) is the second invariant (Q in Zhou et. al. 1999)
    #       q = -det(velGrad) is the third invariant (R in Zhou et. al. 1999)
    # The first invariant (P in Zhou et. al. 1999) is the divergence of velocity, and is zero

    # Sorting out elements of the velocity gradient
    a00=a[:,:,:,0,0]; a01=a[:,:,:,0,1]; a02=a[:,:,:,0,2]
    a10=a[:,:,:,1,0]; a11=a[:,:,:,1,1]; a12=a[:,:,:,1,2]
    a20=a[:,:,:,2,0]; a21=a[:,:,:,2,1]; a22=a[:,:,:,2,2]

    # Second invariant; -0.5* trace(a@a):
    p = -0.5*(   (a00**2 + a01*a10 + a02*a20)
                +(a10*a01 + a11**2 + a12*a21)
                +(a20*a02 + a21*a12 + a22**2)   )

    # Third invariant; -det(a)
    q = -1.*(    a00*(a11*a22-a12*a21)
                -a01*(a10*a22-a12*a20)
                +a02*(a10*a21-a11*a20)  )


    # Now to find roots of the cubic in s, s^3 + ps +q = 0
    # I use Cardano's method (see Wikipedia article)
    # First, a few terms I need later
    cbrt3 = 3.**(1./3.)
    zeta = np.exp(2.j*np.pi/3.)     # a cube root of 1
    zeta2= np.exp(4.j*np.pi/3.)     # and another 
    
    tmp = np.sqrt(( (q**2)/4. + (p**3)/27. ).astype(np.complex) ) 
    u3 = -q/2.  + tmp 
    # In case u3 has any zeros (which aren't good) due to p being 0, 
    u3f = -q/2. - tmp
    failInd = (u3==0.)
    u3[failInd] = u3f[failInd]
    # Now, u3 doesn't have any zeroes (unless both p and q are both zeros, which should't happen)
    # 
    u = u3**(1./3.)
    v = -p/(3.*u)

    cubeRoots = np.zeros( (a.shape[0], a.shape[1], a.shape[2],3), dtype=np.complex)
    cubeRoots[:,:,:,0] = zeta*u  + zeta2*v
    cubeRoots[:,:,:,1] = zeta2*u + zeta*v
    cubeRoots[:,:,:,2] = u + v

    swirl = cubeRoots.imag.max(axis=-1)
    return swirl



