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
#from pseudo.py import chebint


defaultDict = {'Re': 2000.0, 'flowClass':'channel', 'flowState':'turb', 'eddy':False}
aArrDefault = np.linspace(0., 5., 64)
bArrDefault = np.linspace(0., 5., 128)
NDefault = 251

def getDefaultDict():
    return defaultDict.copy()


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
    else: turb = False
    eddy = flowDict.get('eddy',False)
    print("Computing impulse response at tArr, aArr, bArr:",tArr,aArr, bArr)
    print("Flow parameters are (Re,N,eddy,turb):",(Re,N,eddy,turb))
    linInst = ops.linearize(N=N, flowClass='channel',Re=Re,eddy=eddy,turb=turb)

    # Create flowField instances for each t in tArr, one each for response to x, y, and z impulse
    FxList = []; FyList = []; FzList = []
    for t in tArr:
        flowDict.update({'t':t})
        ffx = flowField(aArr, bArr,N, flowDict=flowDict)
        ffy = ffx.copy(); ffz = ffx.copy()
        FxList.append(ffx)
        FyList.append(ffy)
        FzList.append(ffz)

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
    
    # Save each ff instance if fPrefix is supplied;
    #   Append _Fx_txxxx.mat to the prefix
    if fPrefix is not None:
        fPrefix = fPrefix.split('.')[0]    # Get rid of format suffix, if supplied
        for tInd in range(tArr.size):
            t = tArr[tInd]
            ff = FxList[tInd] 
            ff.saveff(fPrefix+"_Fx_t%05d"%(100.*t))
            ff = FyList[tInd] 
            ff.saveff(fPrefix+"_Fy_t%05d"%(100.*t))
            ff = FzList[tInd] 
            ff.saveff(fPrefix+"_Fz_t%05d"%(100.*t))
    
    return {'FxResponse':FxList[0], 'FyResponse':FyList[0],'FzResponse':FzList[1]}

def loadff(fName):
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

    print("Loaded flowField from ",fName)

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
    else:
        x0 = kwargs['x0']; x1 = kwargs['x1']
        xShift = kwargs.get('xShift',0.)
    if not set(('z0','z1')) <= set(kwargs):
        z0 = -0.5*Lz; z1 = 0.5*Lz; 
    else:
        z0 = kwargs['z0']; z1 = kwargs['z1']
    xArr =  xShift + np.linspace(x0, x1, u.shape[0])
    zArr =  zShift + np.linspace(z0, z1, u.shape[0])

    # I could call flowField.swirl(), which returns swirl, u, vorz.
    #   u and vorz can be added from different sets of modes,
    #   but I can't do that for swirl, so
    velGrad = np.zeros(( xArr.size, zArr.size, yArr.size, 3, 3 ))
    warn("THIS ROUTINE IS INCOMPLETE. DO NOT USE IT.")
    if fNameList is not None:
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
        if flowDict is None:
            flowDict= defaultDict
        else:
            assert set(('Re','flowClass', 'flowState','eddy','t')) <= set(flowDict)
        if len(args) > 0 : aArr = args[0]
        else: aArr = aArrDefault 
        if len(args) > 1 : bArr = args[1]
        else: bArr = bArrDefault 
        if len(args) > 2 : N = args[2]
        else: N = NDefault 
       
         
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
        else:
            turbDict = ops.turbMeanChannel(N=N,Re=flowDict['Re'])
            U = turbDict['U']; dU = turbDict['dU']; d2U = turbDict['d2U']

        obj.U = U; obj.dU = dU; obj.d2U = d2U

        warn("flowField.py (V10.6);\n"+
        "'symm' dropped because I got it wrong. Modes now go positive and negative in kx.")

        return obj
        
    
    def __array_finalize__(self,obj):
        if obj is None: return
         
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
    
    def slice(self,aArr=None,bArr=None,N=None):
        """
        Returns a class instance with increased/reduced aArr, bArr, N 
        Call as new_inst = myFlowField.slice(aArr=newArr, bArr=newArr, N=newN)) 
        """
        
        obj = self.copyArray()

        """ THERE MIGHT BE ISSUES WITH ARRAYS NOT BEING CONTIGUOUS.
        IF THAT HAPPENS USE np.ascontiguousarray(arr) WHEREVER THE ERROR SHOWS UP
        """
        if (aArr is not None) or (bArr is not None):
            warn("Slicing aArr and bArr isn't ready yet... Returning original array... ")
        if (N is not None) and (N != self.N):
            Nnew = abs(int(N))
            Nold = self.N
            newInst = flowField(self.aArr.size, self.bArr.size, Nnew,
                        flowDict=self.flowDict)
            for i0 in range(self.aArr.size):
                for i1 in range(self.bArr.size):
                    for i2 in range(3):
                        newInst[i0,i1,i2] = pseudo.chebint(self[i0,i1,i2],newInst.y)
        else:
            newInst = self.copy()
        return newInst
    
    
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

    def toPhysical(self, arr=None, x0=None, x1=None, z0=None, z1=None, N=None, ySpace='cheb', padded=False):
        """
        Get physical fields from spectral
        Inputs:
            arr:    Any array of spectral coefficients for a scalar of shape consistent with self
            keyword arguments:
            x0, x1, z0, z1; all None by default
                If they're not specified, use domain x in [0,Lx], z in [-Lz/2,Lz/2]
                If any of them is specified, truncate domain to [x0,x1] , [z0,z1]
            N (=None):  If set, interpolate to a different wall-normal grid
            padded (=True): pad x-z modes to twice the number to get finer physical field
            ySpace (='cheb'): If 'cheb', keep data on Chebyshev grid
                                If 'linear', interpolate to uniformly spaced points
        Outputs:
            outDict with keyds
                arrPhys:    Physical field for arr
                xArr, yArr, zArr:   arrays for x, y, and z
        """
        if arr is None:
            arr = self.copyArray()[:,:,0]
            warn("arr is not supplied to 'toPhysical', using streamwise velocity...")
        else: assert (arr.shape == (self.aArr.size, self.bArr.size,self.N)) or \
                (arr.shape == (self.aArr.size, self.bArr.size,1,self.N))
        
        warn("Remember to have both positive and negative streamwise wavenumbers, and non-negative spanwise.")
        # x-Modes go 0,1,..,L-1,L,-L+1,-L+2,..,-1, a total of 2*L
        # z-Modes go 0,1,..,M-1,M, a total of M+1
        L = self.aArr.size//2; M = self.bArr.size-1 
        if padded: nx = 4*L; nz = 4*M
        else: nx = 2*L; nz = 2*M

        # fundamental wavenumbers to define periodic domain
        a0 = self.aArr[0]; b0 = self.bArr[0]
        if a0 == 0.: a0 = self.aArr[1]
        if b0 == 0.: b0 = self.bArr[1]

        # Ensure aArr and bArr are integral multiples
        if not (self.aArr % a0 == 0.).all():
            print("aArr doesn't seem to be integral multiples. Have a look")
            print("a0 is", a0)
            print("aArr/a0 is ", self.aArr/a0)
        if not (self.bArr % b0 == 0.).all():
            print("bArr doesn't seem to be integral multiples. Have a look")
            print("b0 is", b0)
            print("bArr/b0 is ", self.bArr/b0)
       
        # Grids in x, z, and y
        # Worry about x0, x1, z0, z1 after the iFFT
        Lx = 2.*np.pi/a0; Lz = 2.*np.pi/b0
        xArr = np.linspace(0., Lx, nx+1)[:-1]
        zArr = np.linspace(-Lz/2., Lz/2., nz+1)[:-1]
        interpFlag = True
        if N is not None:
            if ySpace == 'linear':
                yArr = np.linspace(1., -1., N+2)[1:-1]
            else: yArr = pseudo.chebdif(N,1)[0]
        else:
            if ySpace == 'linear':
                yArr = np.linspace(1., -1., self.N+2)[1:-1]
            else:
                yArr = self.y.copy()
                interpFlag = False

        
        # Let's start working on truncating the domain according to x0,x1,z0,z1
        if (x0 is None) or (x0<0.): x0ind = 0
        else:   x0ind = np.where(xArr <= x0)[0][-1]     # Index of largest entry of xArr <= x0
        if (x1 is None) or (x1>xArr[-1]): x1ind = nx
        else:   x1ind = np.where(xArr >= x1)[0][0] + 1   # Index of smallest entry of xArr >= x1
        
        if (z0 is None) or (z0< -Lz/2.): z0ind = 0
        else:   z0ind = np.where(zArr <= z0)[0][-1]     # See above
        if (z1 is None) or (z1> zArr[-1]): z1ind = nz
        else:   z1ind = np.where(zArr >= z0)[0][0] + 1
        
        nx1 = x1ind - x0ind
        nz1 = z1ind - z0ind
        xArr = xArr[x0ind:x1ind]; zArr = zArr[z0ind:z1ind]

        arrPhys = _spec2physIfft( arr, padded=padded )[x0ind:x1ind, z0ind:z1ind]
        arrPhys *= (1./(2.*np.pi)**2) * (a0*b0)
        if interpFlag:
            for i0 in range(nx1):
                for i1 in range(nz1):
                    arrPhys[i0,i1] = pseudo.chebint(arr[i0,i1], yArr)

        return {'arrPhys':arrPhys, 'xArr':xArr, 'zArr':zArr, 'yArr':yArr}



    def swirl(self, **kwargs):
        """ Returns the swirling strength for the field in physical space
        IMPORTANT: aArr and bArr must be (positive) integral multiples of aArr[0] and bArr[0]
        This function runs only on numpy's ifft; the custom ifft is now dropped
        Inputs:             
            **kwargs; all of them are passed to flowField.toPhysical(). See its docstring
        Outputs:
            swirlDict with keys 
                swirl, xArr, yArr, zArr
        """
        
        tmpArr = self.ddx()
        ux = tmpArr[:,:,0]
        vx = tmpArr[:,:,1]
        wx = tmpArr[:,:,2]

        uxDict = _spec2physIfft( ux, **kwargs)
        xArr = uxDict['xArr']; yArr = uxDict['yArr']; zArr = uxDict['zArr']
        uxPhys = uxDict['arrPhys']
        # The shapes of velGrad and swirlStrength depend on parameters  x0,x1,z0,z1 in **kwargs
        velGrad = np.zeros( (uxPhys.shape[0], uxPhys.shape[1], uxPhys.shape[2], 3,3) )
        velGrad[:,:,:,0,0] = uxPhys 
        velGrad[:,:,:,0,1] = _spec2physIfft( uy, **kwargs)['arrPhys']
        velGrad[:,:,:,0,2] = _spec2physIfft( uz, **kwargs)['arrPhys']        
        ux = None; uy = None; uz = None # Just in case ux and others aren't just pointers 

        tmpArr = self.ddy()
        uy = tmpArr[:,:,0]
        vy = tmpArr[:,:,1]
        wy = tmpArr[:,:,2]
        velGrad[:,:,:,1,0] = _spec2physIfft( vx, **kwargs)['arrPhys']
        velGrad[:,:,:,1,1] = _spec2physIfft( vy, **kwargs)['arrPhys']
        velGrad[:,:,:,1,2] = _spec2physIfft( vz, **kwargs)['arrPhys']
        vx = None; vy = None; vz = None

        tmpArr = self.ddz()
        uz = tmpArr[:,:,0]
        vz = tmpArr[:,:,1]
        wz = tmpArr[:,:,2]
        velGrad[:,:,:,2,0] = _spec2physIfft( wx, **kwargs)['arrPhys']
        velGrad[:,:,:,2,1] = _spec2physIfft( wy, **kwargs)['arrPhys']
        velGrad[:,:,:,2,2] = _spec2physIfft( wz, **kwargs)['arrPhys']
        wx = None; wy = None; wz = None
        tmpArr = None

        # Ifft needs to be scaled by this factor to kinda sorta account for using a finite number of Fourier modes
        velGrad *= (1./(2.*np.pi)**2) * (a0 * b0)
        
        swirlStrength = velGrad2swirl(velGrad)
        
        return {'swirl':swirlStrength, 'xArr':xArr, 'yArr':yArr, 'zArr':zArr} 

    def savePhysical(self, fieldList=['u'], fName=None, **kwargs):
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
        if fName is None: fName = 'testPhysFields.mat'

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
        else:
            saveDict.update({'xArr':physDict['xArr'], 'yArr':physDict['yArr'], 'zArr':physDict['zArr']})
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
            warn("I'm not checking for coinciding bArr; ensure it doesn't happen.")
            warn("Ensure fields are appended in kz as 0,1,..,M")
            warn("VERY IMPORTANT: ORDERING OF kz in .appendField() is extremely important")
            
            bNew = np.concatenate(( self.bArr.flatten(), ff.bArr.flatten() )) 

            ffLong = flowField( self.aArr, bNew, self.N, flowDict = self.flowDict )
            ffLong[:, :self.bArr.size] = self
            ffLong[:, self.bArr.size:] = ff
        
        elif (self.bArr.size == ff.bArr.size) and (self.bArr == ff.bArr).all():
            #assert not (self.aArr == ff.aArr).any()
            warn("I'm not checking for coinciding aArr; ensure it doesn't happen.")
            warn("Ensure fields are appended in kx as 0,1,..,L,-L+1,..,-1")
            warn("VERY IMPORTANT: ORDERING OF kx in .appendField() is extremely important")
            
            aNew = np.concatenate(( self.aArr.flatten(), ff.aArr.flatten() )) 

            ffLong = flowField( aNew, self.bArr, self.N, flowDict = self.flowDict )
            ffLong[:self.aArr.size] = self
            ffLong[self.aArr.size:] = ff
        else:
            self.messyAppendField(ff)
        return ffLong
        
    def messyAppendField(self,ff):
        """ Combine flowFields, like appendField(), but without the restriction
        When (a,b) in the new ff instance is not in either self or ff, leave zeros

        Inputs:
            ff: flowField to be appended
        Outputs:
            ffLong:     Appended field"""
        raise RuntimeError("This isn't ready yet... Ensure appendField can be used..")
        return



def _spec2physIfft(arr, padded=False):
    # symm decides how I should extend the Fourier coeffs in the first quadrant
    #   of the a-b plane to the second quadrant.
    # For even symm, I extend as f_{-a,b} = conj( f_{a,-b} ) = conj( f_{a,b} )
    # For odd  symm, I extend as f_{-a,b} = conj( f_{a,-b} ) =-conj( f_{a,b} )
    L = arr.shape[0]//2; M = arr.shape[1]-1; N= arr.shape[2]
    warn("Assuming that modes go positive and negative in kx")

    # The array is now ready to be used for numpy's rifft2
    if padded: nx = 4*L; nz = 4*M
    else: nx = 2*L; nz = 2*M

    scaleFactor = nx * nz    # times something related to a0,b0??
    physField =  scaleFactor * np.fft.irfft2( arr, s=(nx, nz),axes=(0,1) )  
    
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


