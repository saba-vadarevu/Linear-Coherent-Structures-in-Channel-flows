""" flowField.py Defines a class (inheriting numpy.ndarray) for plane channel and Couette flows
    Discretization is Fourier, Fourier, Chebyshev (collocation), Fourier in x,y,z
    z is spanwise
Class instances have shape (nx, nz, N) for x,z, and y
    For Fourier discretization, a rectangular domain is used and positive and negative 
        mode coefficients are stored, as ,0,1,...L,-L+1,..,-1 for time with nx=2L, 
        similarly for z
Class attributes are:
    flowDict, N,  y, D, D2, aArr, bArr, U, dU, d2U
        aArr and bArr need not have integral multiples, any set is fine
        ensure aArr >= 0 and bArr >= 0 (do I keep b=0?)
    flowDict has keys:
        Re :Reynolds number based on normalization of U
        flowClass: 'channel', 'couette', or 'bl'
        flowState: 'lam' or 'turb'
        eddy: True/False

Class methods are:
    ddx, ddy, ddz, ddx2, ddy2, ddz2 :  Derivatives in x, y, z; z is spanwise
    div:    Divergence
    velGrad:    Gradient (tensor) of velocity vector
    invariants: P,Q,R of velocity gradient
    swirlStrength: Field of imaginary components of complex eigvals
                    of velocitygradient at each point
    toPhysical: Print (or save to file) physical field

Module functions outside the class:
impulseResponse:    Impulse response for specified modes stored in flowField
"""

""" #####################################################
Sabarish Vadarevu
Melbourne School of Engineering
University of Melbourne, Australia 
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
        warn("You have to supply fPrefix, or the fields aren't going to be saved")
    else :
        fPrefix0 = kwargs.pop('fPrefix')
        fFlag = True

    for aInd in range(na):
        aArr = aArrFull[ aInd*aStep : (aInd+1)*aStep ]
        if (not (na==1)) and fFlag:
            aPrefix = '_aPart%d_%d'%(aInd+1,na)
        else : 
            aPrefix = ''

        for bInd in range(nb):
            bArr = bArrFull[bInd*bStep : (bInd+1)*bStep]
            if (not (nb==1)) and fFlag:
                bPrefix = '_bPart%d_%d'%(bInd+1,nb)
            else :
                bPrefix = ''
            fPrefix = fPrefix0 + aPrefix + bPrefix 
            impulseDict = impulseResponse(aArr, bArr, N, tArr, fPrefix= fPrefix,**kwargs)

    return impulseDict

            

    

def impulseResponse(aArr, bArr,N, tArr, flowDict=defaultDict, impulseArgs=None, fPrefix=None):
    """
    Generate flowField over set of Fourier modes as response to impulse
    Inputs:
        aArr:   Set of streamwise wavenumbers (>=0)
        bArr:   Set of spanwise wavenumbers (>=0). 
                    These are extended to cover the other 3 quadrants in a-b plane
                        when printing physical fields
        t:      Time (non-dimensionalized by U-normalization and channel half height)
        flowDict (=defaultDict):
                Contains keys:
                Re (=2000):     Reynolds number
                flowClass (='channel'):  'channel'/'couette'/'bl'
                flowState (='turb')   :  'lam'/'turb'
                eddy (=False):  True/False
        impulseArgs (=None):
                see kwarg impulseArgs to impulseResponse.timeMap()

    """
    assert (bArr >= 0).all()
    tArr = np.array([tArr]).flatten()
    assert (tArr > 0.).all()
    warn("Need to write some check for wall-normal grid independence")

    if (fPrefix is None):
        tArr = np.array([tArr[0]]).flatten()
        print("No save name is specified. Can return flowFields at just the first 't', so not computing later times.")
        print("Specify kwarg 'fPrefix' to allow computation at multiple times")

    Re = flowDict.get('Re',2000.)
    if flowDict.get('flowState','turb') == 'turb':
        turb = True
    else : turb = False
    eddy = flowDict.get('eddy',False)
    #print("Computing impulse response at tArr, aArr, bArr:",tArr,aArr, bArr)
    print("Flow parameters are (Re,N,eddy,turb):",(Re,N,eddy,turb))
    linInst = ops.linearize(N=N, flowClass='channel',Re=Re,eddy=eddy,turb=turb)

    # Create flowField instances for each t in tArr, one each for response to x, y, and z impulse
    FxList = []; FyList = []; FzList = []; FxyzList = []
    for t in tArr:
        flowDict.update({'t':t})
        ffx = flowField(aArr, bArr,N, flowDict=flowDict)
        ffy = ffx.copy(); ffz = ffx.copy(); ffxyz = ffx.copy()
        FxList.append(ffx)
        FyList.append(ffy)
        FzList.append(ffz)
        FxyzList.append(ffxyz)

    for i0 in range(aArr.size):
        a = aArr[i0]
        print("a:",a)
        for i1 in range(bArr.size):
            b = bArr[i1]
            if (a==0.) and (b==0.): 
                warn("Remember to verify that the (0,0) mode has zero coefficients.")
                continue            
            responseDict = impres.timeMap(a,b,tArr=tArr,linInst=linInst,
                    eddy=eddy,impulseArgs=impulseArgs)
            for tInd in range(tArr.size):
                ff = FxList[tInd]
                ff[i0,i1] = responseDict['coeffArr'][tInd,0].reshape((3,N))
                ff = FyList[tInd]
                ff[i0,i1] = responseDict['coeffArr'][tInd,1].reshape((3,N))
                ff = FzList[tInd]
                ff[i0,i1] = responseDict['coeffArr'][tInd,2].reshape((3,N))
                ff = FxyzList[tInd]
                ff[i0,i1] = responseDict['coeffArr'][tInd,3].reshape((3,N))
    
    # Save each ff instance if fPrefix is supplied;
    #   Append _Fx_txxxx.mat to the prefix
    if fPrefix is not None :
        fPrefix = fPrefix.split('.')[0]    # Get rid of format suffix, if supplied
        for tInd in range(tArr.size):
            t = tArr[tInd]
            ff = FxList[tInd] 
            ff.saveff(fPrefix+"_Fx_t%05d"%(round(100.*t)))
            ff = FyList[tInd] 
            ff.saveff(fPrefix+"_Fy_t%05d"%(round(100.*t)))
            ff = FzList[tInd] 
            ff.saveff(fPrefix+"_Fz_t%05d"%(round(100.*t)))
            ff = FxyzList[tInd]
            ff.saveff(fPrefix+"_Fxyz_t%05d"%(round(100.*t)))
    
    return {'FxResponse':FxList[0], 'FyResponse':FyList[0],'FzResponse':FzList[0],'FxyzResponse':FxyzList[0]}

def loadff(fName,printOut=False):
    if not fName.endswith('.mat'):
        fNamePrefix = fName.split('.')[0]
        fName = fNamePrefix +'.mat'

    loadDict =  loadmat(fName)
    # Because loadmat formats scalars as arrays, 
    flowDict = {'N': int(loadDict['N']), 'Re': float(loadDict['Re']), 't':float(loadDict['t']),\
            'flowClass': str(loadDict['flowClass'][0]), 'flowState': str(loadDict['flowState'][0]), 'eddy': str(loadDict['eddy'][0]) }

    aArr = loadDict['aArr'].flatten(); bArr = loadDict['bArr'].flatten() 
    N = flowDict['N']

    ff = flowField(aArr, bArr, N, flowDict=flowDict)
    ff[:] = loadDict['ffArr']

    if printOut:
        print("Loaded flowField from ",fName)

    return ff

def loadff_split(fPrefix, t, forcing='Fz', na=32, nb=1, **kwargs):
    """
    Load flowField from a set of  .mat files 
    Current convention for naming split flowfield files is
        fPrefix + '_aPart3_32' + '_Fz' + '_txxxxx.mat'  # when only aArr is split but not bArr
        fPrefix + '_bPart5_16' + '_Fx' + '_txxxxx.mat'  # when only bArr is split
        fPrefix + '_aPart3_32' + '_bPart5_16'+ '_Fy' + '_txxxxx.mat'
        t is stored in 5 digits, with last 2 digits representing decimals
    Inputs:
        fPrefix: Prefix that identifies the case, such as 'ffEddyRe10000'
        t: time (float)
        forcing (='Fz'):    Forcing direction
        na (=32):   Number of parts for aArr
        nb (=1) :   Number of parts for bArr
    Outputs:
        flowField instance 
    """
    fSuffix = '_%s_t%05d.mat'%(forcing, round(100*t) )
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
    
def add2physical(u=None, vorz=None, swirl=None, ffList=None, fNameList=None, ySpace='cheb',**kwargs):
    """ 
    Add u, omega_z, and swirling strength due to modes in a supplied flowField 
    Inputs:
        v(=None):  3D array to add 'u' to
        vorz(=None): 3D array to add omega_z to
        swirl(=None):3D array to add swirling strength to
        ff (=None): flowField instance to generate the above fields
        fName (=None):  If ff is not supplied, load ff from file fName
        ySpace (='linear'): If 'linear', use linear grid in y, if 'cheb', use Chebyshev
        **kwargs:   physical grid paramters: x0, x1, z0, z1 
                xArr is generated as np.linspace(x0,x1,nx), similarly zArr
    """
    assert u.shape == vorz.shape == swirl.shape

    if np.linalg.norm(swirl, ord='fro') >= 1.0e-9:
        warn("swirling strength isn't supposed to be superposed\n"+
                "Or maybe it can be, but I don't have proof of that yet.\n"+
                "Anyway, until I don't prove it, don't use superpositon.")

    Lx = 2.*np.pi/ff.aArr[0]; Lz = 2.*np.pi/ff.bArr[0]
    if not set(('x0','x1')) <= set(kwargs):
        x0 = -0.1*Lx; x1 = 0.9*Lx; 
        xShift = kwargs.get('xShift',2./3.* np.amax(ff.U)*ff.flowDict['t'])
    else :
        x0 = kwargs['x0']; x1 = kwargs['x1']
        xShift = kwargs.get('xShift',0.)
    if not set(('z0','z1')) <= set(kwargs):
        z0 = -0.5*Lz; z1 = 0.5*Lz; 
    else :
        z0 = kwargs['z0']; z1 = kwargs['z1']
    xArr =  xShift + np.linspace(x0, x1, u.shape[0])
    zArr =  zShift + np.linspace(z0, z1, u.shape[0])

    # I could call flowField.swirl(), which returns swirl, u, vorz.
    #   u and vorz can be added from different sets of modes,
    #   but I can't do that for swirl, so
    velGrad = np.zeros(( xArr.size, zArr.size, yArr.size, 3, 3 ))
    warn("THIS ROUTINE IS INCOMPLETE. DO NOT USE IT.")
    if fNameList is not None :
        for fName in fNameList:
            ff = loadff(fName)
            u[:] += ff.toPhysical(arr=ff[:,:,0], xArr=xArr, zArr=zArr, N=u.shape[-1], 
                    fName=None, symm='even', ySpace=kwargs['ySpace'])
            ffx = self.ddx() 
            vorz[:] += toPhysical(arr=ff[:,:,0], xArr=xArr, zArr=zArr, N=u.shape[-1], 
                    fName=None, symm='even', ySpace=kwargs['ySpace'])

            print("Successfully added u, vorz, and velGrad from ", fName)

    return





class flowField(np.ndarray):
    """
    Defines a class (inheriting numpy.ndarray) for plane channel and Couette flows
        Discretization is Fourier, Fourier, Chebyshev (collocation), Fourier in x,y,z
        z is spanwise
    Class instances have shape (nx, nz, N) for x,z, and y
        For Fourier discretization, a rectangular domain is used and positive and negative 
            mode coefficients are stored, as ,0,1,...L,-L+1,..,-1 for time with nx=2L, 
            similarly for z
    Class attributes are:
        flowDict, N,  y, D, D1, D2, aArr, bArr, U, dU, d2U, weightDict
            aArr and bArr need not have integral multiples, any set is fine
            ensure aArr >= 0 and bArr >= 0 (do I keep b=0?)
        flowDict has keys:
            Re :Reynolds number based on normalization of U
            flowClass: 'channel', 'couette', or 'bl'
            flowState: 'lam' or 'turb'
            eddy: True/False
            t:      Time at which flowField is supposedly a response to impulse
    
    Methods: 
        verify 
        ddx, ddx2, ddz, ddz2, ddy, ddy2 
        div, curl, toPhysical, swirl
        dot, norm, weighted
        flux, dissipation, energy, powerInput


    self.verify() ensures that the shape attributes are self-consistent. 

    Initialization:
        flowField() creates an instance using a default dictionary: a 3 component zero-vector of shape (64,128,3,251) for turbulent channel flow at Re=2000 
        flowField(aArr, bArr,N,flowDict=dictName) creates an instance with shape (aArr.size, bArr.size,3,N), with flowDict to specify the flow conditions.
    A warning message is printed when the default dictionary is used.
            
    """
    def __new__(cls, *args, flowDict=None,**kwargs):
        """Creates a new instance of flowField class, call as
            flowField(aArr, bArr, N ,flowDict=flowDict)
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
                
        
        obj.flowDict = flowDict.copy()
        obj.aArr = aArr
        obj.bArr = bArr
        obj.N = N
        
        y,DM = pseudo.chebdif(N,2)
        D1 = np.ascontiguousarray(DM[:,:,0]) 
        D2 = np.ascontiguousarray(DM[:,:,1])
        obj.y = y; obj.D1 = D1; obj.D2 = D2; obj.D = D1
       
        obj.weightDict = pseudo.weightMats(N)

        assert flowDict['flowClass'] not in ['couette','bl'], "Currently only channel is supported"
        if flowDict['flowState'] == 'lam':
            U = 1. - y**2; dU = -2.*y; d2U = -2.*np.ones(N)
        else :
            turbDict = ops.turbMeanChannel(N=N,Re=flowDict['Re'])
            U = turbDict['U']; dU = turbDict['dU']; d2U = turbDict['d2U']

        obj.U = U; obj.dU = dU; obj.d2U = d2U

        return obj
        
    
    def __array_finalize__(self,obj):
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

    def saveff(self, fName):
        """ Save flowField instance to .mat file"""
        if not fName.endswith('.mat'):
            fNamePrefix = fName.split('.')[0]
            fName = fNamePrefix +'.mat'
        saveDict = {'aArr':self.aArr, 'bArr':self.bArr, 'N':self.N, 'U':self.U, 'ffArr':self.copyArray()}
        saveDict.update(self.flowDict)

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
            by integrating {self[nd=j]*vec2[nd=j].conj()} along x_j, and adding the integrals for j=1,..,self.nd.
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







    def toPhysical(self, arr=None, x0=None, lx=None, z0=None, N=None, ySpace='cheb', doSort=False, **kwargs):
        """
        Get physical fields from spectral
        Inputs:
            arr:    Any array of spectral coefficients for a scalar of shape consistent with self
            keyword arguments:
            x0, lx, z0; all None by default
                If they're not specified, use domain x in [0,Lx], z in [-Lz/2,Lz/2]
                If any of them is specified, truncate domain to [x0,x0+lx] , [z0,-z0]
                Allow x0 to go in [-Lx/2., 0] if needed, coz sometimes the full structure has some back-propogation
            N (=None):  If set to a value other than self.N, interpolate to a different wall-normal grid
            ySpace (='cheb'): If 'cheb', keep data on Chebyshev grid
                                If 'linear', interpolate to uniformly spaced points
            doSort (=True):  If True, call self.sortWavenumbers()
            **kwargs: accept keys 'L' and 'M' to use for padding
                    sends **kwargs directly to _spec2physIfft() directly to avoid issues with L and M in the current function
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
            warn("Ensure input arr confirms to fft order for wavenumbers")

       
        #===============================================================
        # Ensure fft order for self.aArr and self.bArr 
        #================================
        # x-Modes go 0,1,..,L-1,L,-L+1,-L+2,..,-1, a total of 2*L
        # z-Modes go 0,1,..,M-1,M, a total of M+1
        L0 = self.aArr.size//2; M0 = self.bArr.size-1 
        if ('L' not in kwargs) or (kwargs['L'] is None): L = L0
        else : L = kwargs['L']
        if ('M' not in kwargs) or (kwargs['M'] is None): M = M0
        else : M = kwargs['M']
        #pdb.set_trace()
        # fundamental wavenumbers to define periodic domain
        a0 = np.amin( self.aArr[ np.where(self.aArr > 0.)[0]] )   # Smallest positive wavenumber
        b0 = np.amin( self.bArr[ np.where(self.bArr > 0.)[0]] )   # Smallest positive wavenumber

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
        nx = 2*L; nz = 2*M
        Lx = 2.*np.pi/a0; Lz = 2.*np.pi/b0
        xArr = np.linspace(0., Lx, nx+1)[:-1]
        zArr = np.linspace(-Lz/2., Lz/2., nz+1)[:-1]

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

       
        #==============================================================
        # Get physical field, and re-order according to x0ind and x1ind 
        #===============

        # Treatment in z is quite straight-forward, so go with 
        arrPhysUnfolded = _spec2physIfft( arr, **kwargs)[:, z0ind:z1ind]
        zArr = zArr[z0ind:z1ind]
        nz1 = zArr.size

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
        # Divide by size of array, interpolate if needed
        #======================

        arrPhys *= (1./(2.*np.pi)**2) * (a0*b0)
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
        if np.amax(self.aArr) > -np.amin(self.aArr):
            # This is probably an old case where +L*a0 was kept instead of -L*a0
            # change things up a bit
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

        # Now, aArr should look like aArrIdeal, and bArr as bArrIdeal
        if  not (np.linalg.norm(aArr[aInd] - aArrIdeal) < 1.e-09*a0) :
            print("aArr's elements aren't integral multiples of a0; aArrSorted/a0:", aArr[aInd]/a0)
        if  not (np.linalg.norm(bArr[bInd] - bArrIdeal) < 1.e-09*b0) :
            print("bArr's elements aren't integral multiples of b0; bArrSorted/b0:", bArr[bInd]/b0 )

        # Now, aArr[aInd] and bArr[bInd] should be properly sorted.
        # Get them into fft order
        self.aArr = aArr[np.fft.ifftshift(aInd)]
        self.bArr = bArr[bInd]  # bArr goes 0 to M only
        
        self[:] = self[:,bInd]
        self[:] = self[np.fft.ifftshift(aInd)]
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
        else : warn("Are you sure wavenumbers are in fft order?")
        
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
        
        swirlStrength = velGrad2swirl(velGrad)
        
        return {'swirl':swirlStrength, 'xArr':xArr, 'yArr':yArr, 'zArr':zArr} 

    def savePhysical(self, fieldList=['u'], fName=None,fPrefix=None,forcing='Fz', **kwargs):
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
                fName = fPrefix + '_%s_t%05d.mat'%(forcing, round(100*self.flowDict['t']))

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
        
        if len(saveDict.keys()) == 0:
            warn("Looks like there were no acceptable strings in fieldList:"+str(fieldList))
        else :
            saveDict.update({'xArr':physDict['xArr'], 'yArr':physDict['yArr'], 'zArr':physDict['zArr']})
            saveDict.update(self.flowDict)
            savemat(fName, saveDict)
            print("Saved fields   %s   to file %s"%(str(savedList), fName) )
        
        return



    def zero(self):
        """Returns an object of the same class and shape as self, but with zeros as entries"""
        obj = self.copy()
        obj[:] = 0.
        return obj


    def identity(self):
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
        self.aArr = np.sort(self.aArr); self.bArr = np.sort(self.bArr)
        ff.aArr = np.sort(ff.aArr); ff.bArr = np.sort(ff.bArr)

        assert all([self.flowDict[key] == ff.flowDict[key] for key in self.flowDict])
        assert self.N == ff.N 

        if (self.aArr.size == ff.aArr.size) and (self.aArr == ff.aArr).all():
            #assert not (self.bArr == ff.bArr).any()
            
            bNew = np.concatenate(( self.bArr.flatten(), ff.bArr.flatten() )) 

            ffLong = flowField( self.aArr, bNew, self.N, flowDict = self.flowDict )
            ffLong[:, :self.bArr.size] = self
            ffLong[:, self.bArr.size:] = ff
        
        elif (self.bArr.size == ff.bArr.size) and (self.bArr == ff.bArr).all():
            #assert not (self.aArr == ff.aArr).any()
            
            aNew = np.concatenate(( self.aArr.flatten(), ff.aArr.flatten() )) 

            ffLong = flowField( aNew, self.bArr, self.N, flowDict = self.flowDict )
            ffLong[:self.aArr.size] = self
            ffLong[self.aArr.size:] = ff
        else :
            self.messyAppendField(ff)
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

    def messyAppendField(self,ff):
        """ Combine flowFields, like appendField(), but without the restriction
        When (a,b) in the new ff instance is not in either self or ff, leave zeros

        Inputs:
            ff: flowField to be appended
        Outputs:
            ffLong:     Appended field"""
        raise RuntimeError("This isn't ready yet... Ensure appendField can be used..")
        return

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
            L (=None):      If supplied, and different from self.aArr//2, slice flowField
                                Used for padding, since numpy's padding seems a bit funny
            M (=None):      Same as above, but for spanwise modes
    """
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

def velGrad2swirl(velGrad):
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


