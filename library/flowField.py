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
    assert (aArr >= 0).all() and (bArr >= 0).all()
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
            ff.saveff(fPrefix+"_Fx_t%04d"%(10.*t))
            ff = FyList[tInd] 
            ff.saveff(fPrefix+"_Fy_t%04d"%(10.*t))
            ff = FzList[tInd] 
            ff.saveff(fPrefix+"_Fz_t%04d"%(10.*t))
    
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
    
def add2physical(u=None, vorz=None, swirl=None, ffList=None, fNameList=None, ySpace='linear',**kwargs):
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
        div, grad, curl, invariants, swirl
        dot, norm, weighted
        flux, dissipation, energy, powerInput


    self.verify() ensures that the shape attributes are self-consistent. 

    Initialization:
        flowField() creates an instance using a default dictionary: a 3 component zero-vector of shape (64,128,3,251) for turbulent channel flow at Re=2000 
        flowField(aArr, bArr,N,flowDict=dictName) creates an instance with shape (aArr.size, bArr.size,3,N), with flowDict to specify the flow conditions.
    A warning message is printed when the default dictionary is used.
            
    """
    def __new__(cls, *args, flowDict=None,**kwargs):
        """Creates a new instance of flowField class with arguments (cls, arr=None,flowDict=None,dictFile=None)
        cls argument can be used to initialize subclasses of flowField: flowFieldWavy or flowFieldRiblet
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
        aArr = np.sort(aArr)
        bArr = np.sort(bArr)
        
        obj = np.ndarray.__new__(cls,
                shape=(aArr.size,bArr.size,3,N),dtype=np.complex)
                
        
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

        warn("flowField.py (V10.2);\n"+
        "Need to clean up old get Physical functions.\n" +
        "Need to integrate swirl() with toPhysical().\n"+
        "Think about using fft when creating physical fields.")

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

    def toPhysical(self, arr=None, xArr=None, zArr=None, N=None, fName=None, symm='even', ySpace='linear'):
        if arr is None:
            arr = self.copyArray()[:,:,0]
            warn("arr is not supplied to 'toPhysical', using streamwise velocity...")
        else: assert arr.shape == (self.aArr.size, self.bArr.size, 1, self.N)
        xShift = np.amax(self.U) * self.flowDict['t']
        if xArr is None:
            a0 = self.aArr[0]
            if a0 == 0.: a0 = self.aArr[1]
            Lx = 2.*np.pi/a0
            xArr = xShift + np.linspace(0., Lx, 64)
            print("xArr is not supplied... Using 64 points in %.3g + [0,%.3g]"%(xShift,Lx))
        else:
            assert xArr.ndim == 1
            if not ( xArr[0] <= xShift <= xArr[-1] ):
                print("xArr doesn't contain xShift = %.3g.. You sure it's okay?"%xShift)
        if zArr is None:
            b0 = self.bArr[0]
            if b0 == 0.: b0 = self.bArr[1]
            Lz = 2.*np.pi/b0
            zArr = np.linspace(-Lz/2, Lz,64) 
            print("zArr is not supplied... Using 128 points in [%.3g,%.3g]"%(-Lz/2, Lz/2))
        else:
            assert zArr.ndim ==1
        if N is None:
            N = self.N
        if ySpace == 'linear':
            yArr = np.linspace(1., -1., N+2)[1:-1]
            interpFlag = True
        elif N == self.N: 
            yArr = self.y; interpFlag = False
        else:
            yArr = pseudo.chebdif(N,1)[0]
            interpFlag = True

        if self.aArr[0] == 0:
            warn("Ignoring all a=0 modes in toPhysical.")
            arr[0] = 0.
        if self.bArr[0] == 0:
            warn("Ignoring all b=0 modes in toPhysical.")
            arr[:,0] = 0.

        arrPhys = np.zeros((xArr.size, zArr.size, yArr.size)) 
        def _spec2phys(x,z,someArr,symm='even'):
            # symm = even refers to f(z) = f(-z), 
            #   and the other (symm=odd) refers to f(z) = -f(-z)
            # Even symm holds for ux, uy, vx, vy, wz
            # Odd symm holds for uz, vz, wx, wy
            if symm == 'even':
                physArr = 4.* np.sum( np.sum(
                    np.cos(bArr * z) * np.real( someArr * np.exp(1.j*aArr*x) ),
                    axis = 1), axis = 0)
            else:
                physArr = -4.* np.sum( np.sum(
                    np.sin(bArr * z) * np.imag( someArr * np.exp(1.j*aArr*x) ),
                    axis = 1), axis = 0)
            return np.real(physArr)
        
        for i0 in range(xArr.size):
            for i1 in range(zArr.size):
                tmpPhys = _spec2phys(xArr[i0],zArr[i1],arr, symm=symm)
                if interpFlag:
                    arrPhys[i0,i1] = pseudo.chebint(tmpPhys, yArr)
                else:
                    arrPhys[i0,i1] = tmpPhys

        if fName is not None:
            if not (fName[-4:] =='.mat'): fName = fName + '.mat'
            flowDict = self.flowDict
            saveDict = {'arrPhys':arrPhys,'xArr':xArr,'zArr':zArr,'yArr':yArr,\
                    'aArr':self.aArr, 'bArr':self.bArr,'N':self.N,\
                    'Re':flowDict['Re'], 'eddy':flowDict['eddy'], 't':flowDict['t'],\
                    'flowState':flowDict['flowState']}
            savemat(fName, saveDict )
        return arrPhys



    def swirl(self, xArr=None, zArr=None, N=None, fName=None, 
            uField=False,vorzField=False, saveff=False, ySpace='linear', fft=False):
        """ Returns the swirling strength for the field in physical space
        Inputs:
            xArr, zArr (=None,None) :  Coordinates for x and z
                If not set, use the smallest non-zero 'a' and 'b'
                    For z, use 0: 2pi/b
                    For x, use self.U[y=0]*self.flowDict['t'] + 0:2pi/a
            N (=None):  If set, interpolate to a different wall-normal grid
            fName (=None): If not None, dump all of the important bits to a .mat file
            uField (=False): If True, return/save streamwise velocity field  (physical)
            vorzField(=False): If True, return/save spanwise vorticity field (physical)
            saveff (=False):  If True, include the array of the spectral flow field in the .mat file
        """
        
        xShift = np.amax(self.U) * self.flowDict['t']
        if xArr is None:
            a0 = self.aArr[0]
            if a0 == 0.: a0 = self.aArr[1]
            Lx = 2.*np.pi/a0
            xArr = xShift + np.linspace(0., Lx, 64)
            print("xArr is not supplied... Using 64 points in %.3g + [0,%.3g]"%(xShift,Lx))
        else:
            assert xArr.ndim == 1
            if not ( xArr[0] <= xShift <= xArr[-1] ):
                print("xArr doesn't contain xShift = %.3g.. You sure it's okay?"%xShift)
        if zArr is None:
            b0 = self.bArr[0]
            if b0 == 0.: b0 = self.bArr[1]
            Lz = 2.*np.pi/b0
            zArr = np.linspace(-Lz/2, Lz,64) 
            print("zArr is not supplied... Using 128 points in [%.3g,%.3g]"%(-Lz/2, Lz/2))
        else:
            assert zArr.ndim ==1
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



        # Including (a,0) and (0,b) is a bit of a pain. For now, I'll ignore them 
        a0bool = (self.aArr[0] == 0.)
        b0bool = (self.bArr[0] == 0.)
        def _seta0b0(someArr):
            if a0bool: someArr[0] = 0.
            if b0bool: someArr[:,0] = 0.
            return
        
        # self has only first quadrant of kx-kz plane
        # To account for the other 3 quadrants, I must multiply by 4
         
        
        tmpArr = self.ddx()
        _seta0b0(tmpArr)     # Get rid of (a,0) and (0,b) modes if they're in there
        ux = tmpArr.copyArray()[:,:,0]
        vx = tmpArr.copyArray()[:,:,1]
        wx = tmpArr.copyArray()[:,:,2]
        tmpArr = self.ddy()
        _seta0b0(tmpArr)     # Get rid of (a,0) and (0,b) modes if they're in there
        uy = tmpArr.copyArray()[:,:,0]
        vy = tmpArr.copyArray()[:,:,1]
        wy = tmpArr.copyArray()[:,:,2]
        tmpArr = self.ddz()
        _seta0b0(tmpArr)     # Get rid of (a,0) and (0,b) modes if they're in there
        uz = tmpArr.copyArray()[:,:,0]
        vz = tmpArr.copyArray()[:,:,1]
        wz = tmpArr.copyArray()[:,:,2]

        if uField:
            uSpec = self.copyArray()[:,:,0]
            _seta0b0(uSpec)
            if not fft:
                uPhys = np.zeros((xArr.size, zArr.size, yArr.size))
                # If doing fft, there's no need to explicitly define uPhys first,
                #   I can assign it to the returned value of _spec2physIfft
        if vorzField:
            vorzSpec = vx - uy
            if not fft:
                vorzPhys = np.zeros((xArr.size, zArr.size, yArr.size))
                # See comments above

        if not fft: 
            swirlStrength = np.zeros((xArr.size, zArr.size, yArr.size))
        else:
            swirlStrength = np.zeros((2*self.shape[0], 2*self.shape[1], yArr.size))

        aArr = self.aArr.reshape((self.aArr.size,1,1))
        bArr = self.bArr.reshape((1,self.bArr.size,1))

        # Extracting physical field at some (x,z) is done as so:
        # It'd be straight-forward if I had all 4 quadrants of a-b, but since I have them
        #   I need to account for spanwise reflectional symmetry.
        def _spec2phys(x,z,arr,symm='even'):
            # symm = even refers to f(z) = f(-z), 
            #   and the other (symm=odd) refers to f(z) = -f(-z)
            # Even symm holds for ux, uy, vx, vy, wz
            # Odd symm holds for uz, vz, wx, wy
            if symm == 'even':
                physArr = 4.* np.sum( np.sum(
                    np.cos(bArr * z) * np.real( arr * np.exp(1.j*aArr*x) ),
                    axis = 1), axis = 0)
            else:
                physArr = -4.* np.sum( np.sum(
                    np.sin(bArr * z) * np.imag( arr * np.exp(1.j*aArr*x) ),
                    axis = 1), axis = 0)
            return np.real(physArr)

        a0 = self.aArr[0]; b0 = self.bArr[0]
        # Just to be sure I'm not messing up, ensure aArr and bArr are integral multiples 
        assert np.linalg.norm( self.aArr - a0*np.arange(1,self.shape[0]+1) ) <= 1.e-09
        assert np.linalg.norm( self.bArr - b0*np.arange(1,self.shape[1]+1) ) <= 1.e-09

        def _spec2physIfft(arr, symm='even'):
            # Let's use ifft to get the field this time
            # symm decides how I should extend the Fourier coeffs in the first quadrant
            #   of the a-b plane to the second quadrant.
            # For even symm, I extend as f_{-a,b} = conj( f_{a,-b} ) = conj( f_{a,b} )
            # For odd  symm, I extend as f_{-a,b} = conj( f_{a,-b} ) =-conj( f_{a,b} )
            L = arr.shape[0]; M = arr.shape[1]
            assert L > 1, "Use _spec2phys() instead of _spec2physIfft() for such small cases"
            arrExt = np.zeros( (2*L, M+1,yArr.size), dtype=np.complex)
            # numpy's ifft needs coeffs for a to go as 0,1,..,L, -L+1,-L+2,...,-1
            #       and for b to go as 0,1,..,M
            arrExt[1:L+1, 1:] = arr
            # Apparently, the largest mode must have real coefficients
            arrExt[L] = np.real(arrExt[L]); arrExt[:,M] = np.real(arrExt[:,M])
            # For L = 3, a goes as [0,1,2,3,-2,-1]
            if symm == 'even':
                arrExt[ :L:-1,1:] = np.conj( arr[ :L-1,: ] ) 
            if symm == 'odd':
                arrExt[ :L:-1,1:] =-np.conj( arr[ :L-1,: ] ) 

            # The array is now ready to be used for numpy's rifft2
            scaleFactor = (2*L * 2*M)   # times something related to a0,b0??
            return scaleFactor*np.fft.irfft2( arrExt, axes=(0,1) )  

        
        # Shouldn't be too hard to figure out a better way,
        #   but I'll stick with the loop for now
        if not fft:
            velGrad = np.zeros((self.N,3,3))
            invariants = np.zeros((self.N,3))
            for i0 in range(xArr.size):
                x = xArr[i0]
                for i1 in range(zArr.size):
                    z = zArr[i1]
                    # ux(z)=ux(-z)
                    velGrad[:,0,0] = _spec2phys(x,z,ux,symm='even')
                    # uy(z)=uy(-z)
                    velGrad[:,0,1] = _spec2phys(x,z,uy,symm='even') + self.dU
                    # uz(z)=-uz(-z)
                    velGrad[:,0,2] = _spec2phys(x,z,uz,symm='odd')
                    
                    # vx(z)=vx(-z)
                    velGrad[:,1,0] = _spec2phys(x,z,vx,symm='even')
                    # vy(z)=vy(-z)
                    velGrad[:,1,1] = _spec2phys(x,z,vy,symm='even')
                    # vz(z)=-vz(-z)
                    velGrad[:,1,2] = _spec2phys(x,z,vz,symm='odd')

                    # wx(z)=-wx(-z)
                    velGrad[:,2,0] = _spec2phys(x,z,wx,symm='odd')
                    # wy(z)=-wy(-z)
                    velGrad[:,2,1] = _spec2phys(x,z,wy,symm='odd')
                    # wz(z)=wz(-z)
                    velGrad[:,2,2] = _spec2phys(x,z,wz,symm='even')

                    if uField:
                        uPhys[i0,i1] = _spec2phys(x,z,uSpec,symm='even')
                        if interpFlag:
                            uPhys[i0,i1] = pseudo.chebint(uPhys[i0,i1],yArr)
                    if vorzField:
                        vorzPhys[i0,i1] = -self.dU + _spec2phys(x,z,vorzSpec, symm='even')
                        if interpFlag:
                            vorzPhys[i0,i1] = pseudo.chebint(vorzPhys[i0,i1], yArr)

                    # We have velGrad now, on to calculating the invariants, P,Q,R
                    # Refer to Chakraborty, Balachandar and Adrian, JFM 2005
                    # Just use eigenvalue calculation instead of solving characteristic equations
                    # Return to this after profiling
                    tmpSwirl = np.zeros(self.N)
                    for i2 in range(self.N):
                        evals = np.linalg.eigvals(velGrad[i2])
                        tmpSwirl[i2] = np.max( np.imag(evals))
                    if interpFlag:
                        swirlStrength[i0,i1] = pseudo.chebint(tmpSwirl, yArr)
                    else:
                        swirlStrength[i0,i1] = tmpSwirl
                    # Is there something about orientation of vortex cores?
        else:
            # This uses lots more memory, but should be faster
            velGrad = np.zeros( (2*self.shape[0], 2*self.shape[1], yArr.size, 3, 3) )
            velGrad[:,:,:,0,0] = _spec2physIfft( ux, symm='even')
            velGrad[:,:,:,0,1] = _spec2physIfft( uy, symm='even')
            velGrad[:,:,:,0,2] = _spec2physIfft( uz, symm='odd' )

            velGrad[:,:,:,1,0] = _spec2physIfft( vx, symm='even')
            velGrad[:,:,:,1,1] = _spec2physIfft( vy, symm='even')
            velGrad[:,:,:,1,2] = _spec2physIfft( vz, symm='odd' )

            velGrad[:,:,:,2,0] = _spec2physIfft( wx, symm='odd' )
            velGrad[:,:,:,2,1] = _spec2physIfft( wy, symm='odd' )
            velGrad[:,:,:,2,2] = _spec2physIfft( wz, symm='even')
            # The velocity gradient tensor is now in memory, let's get the swirling strength
            for i0 in range(velGrad.shape[0]):
                for i1 in range(velGrad.shape[1]):
                    for i2 in range(velGrad.shape[2]):
                        evals = np.linalg.eigvals(velGrad[i0,i1,i2])
                        swirlStrength[i0,i1,i2] = np.max( np.imag(evals) )
                    if interpFlag:
                        swirlStrength[i0,i1] = pseudo.chebint(swirlStrength[i0,i1], yArr)
            velGrad = None

            if uField:
                uPhys = _spec2physIfft( uSpec, symm='even')
                if interpFlag:
                    uPhys[i0,i1] = pseudo.chebint(uPhys[i0,i1],yArr)
            if vorzField:
                vorzPhys[i0,i1] = -self.dU + _spec2phys(x,z,vorzSpec, symm='even')
                if interpFlag:
                    vorzPhys[i0,i1] = pseudo.chebint(vorzPhys[i0,i1], yArr)



        if fName is not None:
            if not (fName[-4:] =='.mat'): fName = fName + '.mat'
            flowDict = self.flowDict
            saveDict = {'swirl':swirlStrength,'xArr':xArr,'zArr':zArr,'yArr':yArr,\
                    'aArr':self.aArr, 'bArr':self.bArr,'N':self.N,\
                    'Re':flowDict['Re'], 'eddy':flowDict['eddy'], 't':flowDict['t'],\
                    'flowState':flowDict['flowState']}
            if uField: saveDict.update({'u':uPhys})
            if vorzField: saveDict.update({'vorz':vorzPhys})
            if saveff: saveDict.update({'ffArr':self.copyArray()})
            savemat(fName, saveDict )
        returnList = [swirlStrength]
        if uField:  returnList.append(uPhys)
        if vorzField: returnList.append(vorzPhys)
        
        return returnList




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
            bNew = np.sort( np.concatenate(( self.bArr, ff.bArr )) )
            ffLong = flowField( self.aArr, bNew, self.N, flowDict = self.flowDict )
            for i0 in range(ffLong.shape[0]):
                a = self.aArr[i0]
                i1self = 0; i1ff = 0
                for i1 in range(ffLong.shape[1]):
                    b = bNew[i1]
                    # This is a little bit tricky 
                    # I don't want to keep finding the index of b in self.bArr or ff.bArr
                    # Since the bArrs in all 3 instances are sorted, just start by pointing
                    #   at bIndex = 0 for all 3, then advance by 1 everytime there's a match
                    if b in self.bArr:
                        ffLong[i0,i1] = self[i0,i1self]
                        i1self += 1
                    elif b in ff.bArr:
                        ffLong[i0,i1] = ff[i0,i1ff]
                        i1ff += 1
                    else:
                        print("Something's wrong with appending at a,b=",a,b)
        
        elif (self.bArr.size == ff.bArr.size) and (self.bArr == ff.bArr).all():
            #assert not (self.aArr == ff.aArr).any()
            warn("I'm not checking for coinciding aArr; ensure it doesn't happen.")
            aNew = np.sort( np.concatenate(( self.aArr.flatten(), ff.aArr )) )
            ffLong = flowField( aNew, self.bArr, self.N, flowDict = self.flowDict )
            for i1 in range(ffLong.shape[1]):
                b = self.bArr[i1]
                i0self = 0; i0ff = 0
                for i0 in range(ffLong.shape[0]):
                    a = aNew[i0]
                    # This is a little bit tricky 
                    # I don't want to keep finding the index of a in self.aArr or ff.aArr
                    # Since the aArrs in all 3 instances are sorted, just start by pointing
                    #   at aIndex = 0 for all 3, then advance by 1 everytime there's a match
                    if a in self.aArr:
                        ffLong[i0,i1] = self[i0self,i1]
                        i0self += 1
                    elif a in ff.aArr:
                        ffLong[i0,i1] = ff[i0ff,i1]
                        i0ff += 1
                    else:
                        print("Something's wrong with appending at a,b=",a,b)
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

