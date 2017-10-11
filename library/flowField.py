""" flowField.py
Defines a class (inheriting numpy.ndarray) for plane channel and Couette flows
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


def impulseResponse(aArr, bArr,N, t, flowDict=defaultDict, impulseArgs=None):
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
    assert t > 0.
    warn("Need to write some check for wall-normal grid independence")
    Re = flowDict.get('Re',2000.)
    if flowDict.get('flowState','turb') == 'turb':
        turb = True
    else: turb = False
    eddy = flowDict.get('eddy',False)
    print("Computing impulse response at t, aArr, bArr:",t,aArr, bArr)
    print("Flow parameters are (Re,N,eddy,turb):",(Re,N,eddy,turb))
    linInst = ops.linearize(N=N, flowClass='channel',Re=Re,eddy=eddy,turb=turb)

    # Create a flowField instance 
    flowDict.update({'t':t})
    ffx = flowField(aArr, bArr,N, flowDict=flowDict)
    ffy = ffx.copy(); ffz = ffx.copy()
    tArr = np.array([t])
    for i0 in range(aArr.size):
        a = aArr[i0]
        print("a:",a)
        for i1 in range(bArr.size):
            b = bArr[i1]
            responseDict = impres.timeMap(a,b,tArr=tArr,linInst=linInst,
                    eddy=eddy,impulseArgs=impulseArgs)
            ffx[i0,i1] = responseDict['coeffArr'][0,0].reshape((3,N))
            ffy[i0,i1] = responseDict['coeffArr'][0,1].reshape((3,N))
            ffz[i0,i1] = responseDict['coeffArr'][0,2].reshape((3,N))
            
    
    return {'FxResponse':ffx, 'FyResponse':ffy,'FzResponse':ffz}





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
        direcDeriv, ifft, getPhysical, makePhysical, makePhysicalPlanar


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
        def _spec2phys(x,z,someArr):
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
            uField=False,vorzField=False, saveff=False):
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
        if N is not None:
            yArr = np.linspace(1., -1., N+2)[1:-1]
        else:
            yArr = np.linspace(1., -1., self.N+2)[1:-1]



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
            uPhys = np.zeros((xArr.size, zArr.size, yArr.size))
        if vorzField:
            vorzSpec = vx - uy
            vorzPhys = np.zeros((xArr.size, zArr.size, yArr.size))
        
        swirlStrength = np.zeros((xArr.size, zArr.size, yArr.size))
        velGrad = np.zeros((self.N,3,3))
        invariants = np.zeros((self.N,3))
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

        def _customDet(arr):
            # compute determinants for 3d array of shape (N,3,3)
            # Determinant of the velocity gradient tensor is the third invariant
            # For matrix [a b c; d e f; g h i], writing det as 
            # a(ei-fh) - b(di - fg) + c (dh -eg)
            det =     arr[:,0,0] * ( arr[:,1,1]*arr[:,2,2] - arr[:,1,2]*arr[:,2,1] ) \
                    - arr[:,0,1] * ( arr[:,1,0]*arr[:,2,2] - arr[:,1,2]*arr[:,2,0] ) \
                    + arr[:,0,2] * ( arr[:,1,0]*arr[:,2,1] - arr[:,1,1]*arr[:,2,0] )
            return det
        
        # Shouldn't be too hard to figure out a better way,
        #   but I'll stick with the loop for now
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
                    uPhys[i0,i1] = self.U + _spec2phys(x,z,uSpec,symm='even')
                if vorzField:
                    vorzPhys[i0,i1] = -self.dU + _spec2phys(x,z,vorzSpec, symm='even')

                # We have velGrad now, on to calculating the invariants, P,Q,R
                # Refer to Chakraborty, Balachandar and Adrian, JFM 2005
                if False:
                    # P = - div([u,v,w]); this should be zero
                    invariants[:,0] = -(velGrad[:,0,0] + velGrad[:,1,1] + velGrad[:,2,2])
                    divSum = np.linalg.norm(invariants[:,0])
                    if divSum > 1.e-06:
                        warn("Divergence at (x,z)=(%.3g,%.3g) is over 10^-6:%.3g..."%(x,z,divSum))
                        warn("Swirl strength calculation assumes zero divergence")

                    # Q = 0.5( trace(Omega@Omega.T) - trace(S@S.T) ), 
                    #   S and Omega are the symmetric and anti-symmetric parts of the velocity gradient tensor
                    symVelGrad = 0.5* (velGrad + np.swapaxes(velGrad, axis1=1,axis2=2) )
                    antiVelGrad = 0.5* (velGrad - np.swapaxes(velGrad, axis1=1,axis2=2) )
                    invariants[:,1] = 0.5*( np.trace(antiVelGrad, axis1=1, axis2=2) - np.trace(symVelGrad, axis1=1, axis2=2) )

                    # R = -det(velGrad)
                    invariants[:,2] = - _customDet(velGrad)

                    # The eigenvalues of the velocity gradient tensor are solutions to the cubic
                    # l^3 + P l^2 + Ql + R = 0, where P, Q, and R are the invariants calculated above
                    # I assume P = 0, as it should be, to reduce the cubic equation to
                    # l^3 + Ql + R = 0
                    # Solutions to this equation are 
                else:
                    # Just use eigenvalue calculation instead of solving characteristic equations
                    # Return to this after profiling
                    tmpSwirl = np.zeros(self.N)
                    for i2 in range(self.N):
                        evals = np.linalg.eigvals(velGrad[i2])
                        tmpSwirl[i2] = np.max( np.imag(evals))
                    swirlStrength[i0,i1] = pseudo.chebint(tmpSwirl, yArr)
                        # Is there something about orientation of vortex cores?
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

                    






        

#====================================================================================
#====================================================================================
#====================================================================================
#====================================================================================
#====================================================================================
#====================================================================================
#====================================================================================
    def getPhysical(self,tLoc = 0., xLoc=0., zLoc= 0., yLoc = None):
        """Returns the flow field at specified locations in t,x,z,y
        Arguments: tLoc (default=0.), xLoc (default=0.), zLoc (default=0.)
                    yLoc (default:None, corresponds to Chebyshev nodes on [1,-1] of cardinality self.N)
                    if yLoc is specified, it must be either 
        NOTE: yLoc here is referenced from the local wall locations. The walls are ALWAYS at +/-1, including for wavy walls"""
        tLoc = np.asarray(tLoc); xLoc = np.asarray(xLoc); zLoc = np.asarray(zLoc)
        # If, say, tLoc was initially a float, the above line converts it into a 0-d numpy array
        # I can't call tLoc as tLoc[0], since 0-d arrays can't be indexed. So, convert them to 1-d:
        tLoc = tLoc.reshape(tLoc.size); xLoc = xLoc.reshape(xLoc.size); zLoc = zLoc.reshape(zLoc.size)
        
        
        # Ensure that Fourier modes for the full x-z plane are available, not just the half-plane (m>=0)
        M = self.flowDict['M']
            
        if yLoc is None:
            field = np.zeros((tLoc.size, xLoc.size, zLoc.size, self.nd, self.N))
            for tn in range(tLoc.size):
                for xn in range(xLoc.size):
                    for zn in range(zLoc.size):
                        field[tn,xn,zn] = self.ifft(tLoc=tLoc[tn], xLoc=xLoc[xn], zLoc=zLoc[zn])
            yLoc = chebdif(self.N,1)[0]
        else:
            yLoc = np.asarray(yLoc).reshape(yLoc.size)
            assert not any(np.abs(yLoc)-1.>1.0e-7), 'yLoc must only have points in [-1,1]'
            field = np.zeros((tLoc.size, xLoc.size, zLoc.size, self.nd, yLoc.size))
            for tn in range(tLoc.size):
                for xn in range(xLoc.size):
                    for zn in range(zLoc.size):
                        fieldTemp = self.ifft(tLoc=tLoc[tn], xLoc=xLoc[xn], zLoc=zLoc[zn])
                        for scal in range(self.nd):
                            field[tn,xn,zn,scal] = chebint(fieldTemp[scal], yLoc)

        return field
            
    
    def printPhysical(self,xLoc=None, zLoc=None, tLoc=None, yLoc=None,yOff=0.,pField=None, interY=2,fName='ff'):
        """Prints the velocities and pressure in a .dat file with columns ordered as Y,Z,X,U,V,W,P
        Arguments (all keyword):
            xLoc: x locations where field variables need to be computed 
                    (default: [0:2*pi/alpha] 40 points in x when alpha != 0., and just 1 (even when xLoc supplied) when alpha == 0.)
            zLoc: z locations where field variables need to be computed 
                    (default: [0:2*pi/beta] 20 points in z when beta != 0., and just 1 (z=0) when beta == 0.)
            tLoc: temporal locations (default: 7 points when omega != 0, 1 when omega = 0). Fields at different time-locations are printed to different files
            pField: Pressure field (computed with divFree=False, nonLinear=True if pField not supplied)
            interY: Field data is interpolated onto interY*self.N points before printing. Default for interY is 2
            yOff: Use this to define wavy surfaces. For flat walls, yOff = 0. For wavy surfaces, yOff = 2*eps
                    yOff is used to modify y-grid as   y[tn,xn,zn] += yOff*cos(alpha*x + beta*z - omega*t)
            yLoc: Use this to specify a y-grid. When no grid is specified, Chebyshev nodes are used
            fname: Name of .dat file to be printed to. Default: ff.dat
        """
        a = self.flowDict['alpha']; b = self.flowDict['beta']; omega = self.flowDict['omega']
        K = self.flowDict['K']; L = self.flowDict['L']; M=-np.abs(self.flowDict['M'])
        if (a==0.): 
            return self.printPhysicalPlanar(etaLoc=zLoc, tLoc=tLoc, yLoc=yLoc,yOff=yOff,pField=pField, interY=interY,toFile=toFile,fName=fName)
        if (b==0.): 
            return self.printPhysicalPlanar(etaLoc=xLoc, tLoc=tLoc, yLoc=yLoc,yOff=yOff,pField=pField, interY=interY,toFile=toFile,fName=fName)
        if xLoc is None:
            xLoc = np.arange(0., 2.*np.pi/a, 2.*np.pi/a/40.)
        if zLoc is None:
            zLoc = np.arange(0., 2.*np.pi/b, 2.*np.pi/b/20.)
        if tLoc is None:
            if omega != 0.: tLoc = np.arange(0,2.*np.pi/omega, 2.*np.pi/b/7.)
            else: tLoc = np.zeros(1)
        if yLoc is None:
            yLoc = chebdif(interY*self.N,1)[0]
            yLocFlag = False
        else:
            yLocFlag = True
            assert isinstance(yLoc,np.ndarray) and (yLoc.ndim == 1), 'yLoc must be a 1D numpy array'
            assert not any(np.abs(yLoc) > 1), 'yLoc must only have points in [-1,1]' 
            
        assert (type(yOff) is np.float) or (type(yOff) is np.float64), 'yOff characterizes surface deformation and must be of type float'
        if '.dat' in fName[-4:]: fName = fName[:-4]
        
        assert self.nd == 3, 'makePhysical() is currently written to handle only 3C velocity fields'
        assert isinstance(xLoc,np.ndarray) and isinstance(zLoc,np.ndarray) and isinstance(tLoc,np.ndarray),\
            'xLoc, zLoc, and tLoc must be numpy arrays'
        assert isinstance(fName,str), 'fName must be a string'
        
        if pField is None: pField = self.solvePressure(divFree=False,nonLinear=True)[0]
        else:
            assert pField.size == self.size//3, 'pField must be the same size of each component of self'
        
        obj = self.appendField(pField)        
        
        if interY != 1 and not yLocFlag:
            obj = obj.slice(N=yLoc.size)
        
        
        dataArr = np.zeros((7,tLoc.size,xLoc.size,zLoc.size,yLoc.size))
        
        # Calculating flow field variables at specified t,x,z,y (refer to .ifft() and .getField())
        if not yLocFlag: fields = obj.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc)
        else:  fields = obj.getPhysical(tLoc=tLoc, zLoc=zLoc, xLoc=xLoc, yLoc=yLoc)
        
        # Writing grid point locations:
        #    In output file, columns are ordered as y,z,x, with fields at different time instances in different files
        tLoc = tLoc.reshape(tLoc.size,1,1,1)   # Numpy broadcasting rules repeat entries when size along an axis is 1
        xLoc = xLoc.reshape(1,xLoc.size,1,1)
        zLoc = zLoc.reshape(1,1,zLoc.size,1)
        yLoc = yLoc.reshape(1,1,1,yLoc.size)
        
        dataArr[0] = yLoc + yOff*np.cos(a*xLoc+b*zLoc-omega*tLoc)
        dataArr[1] = zLoc; dataArr[2] = xLoc 
        
        for scal in range(4):
                dataArr[3+scal] = fields[:,:,:,scal]
            
        variables = 'VARIABLES = "Y", "Z", "X", "U", "V", "W", "P"\n'
        zone = 'ZONE T="", I='+str(yLoc.size)+', J='+str(zLoc.size)+', K='+str(xLoc.size)+', DATAPACKING=POINT'
        if tLoc.size == 1:
            #np.savetxt(fName+'.csv', dataArr.reshape((7,dataArr.size//7)).T,delimiter=',')
            #tempArr = dataArr.reshape(dataArr.size)
            title = 'TITLE= "Flow in wavy walled channel with a='+str(a)+', b='+str(b)+\
                ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
            hdr = title+variables+zone
            np.savetxt(fName+'.dat',dataArr.reshape(7,dataArr.size//7).T, header=hdr,comments='')
            print('Printed physical field to file %s.dat'%fName)
        else:
            for tn in range(tLoc.size):
                title = 'TITLE= "Flow in wavy walled channel at t='+str(tLoc[tn,0,0,0])+' with a='+str(a)+', b='+str(b)+\
                    ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
                hdr = title+variables+zone
                np.savetxt(fName+str(tn)+'.dat', dataArr[:,tn].reshape((7,dataArr[:,tn].size//7)).T,header=hdr,comments='')
            print('Printed %d time-resolved physical fields to files %sX.dat'%(tLoc.size,fName))
        return
    
    def printPhysicalPlanar(self,nLoc=40,etaLoc=None, tLoc=None, yLoc=None,yOff=0.,pField=None, interY=2,fName='ffPlanar'):
        """Prints flowField on the plane beta*x - alpha*z = 0 (this plane has normal (beta,-alpha)),
                in coordinate eta := alpha*x + beta*z
        nLoc: Number of wall-parallel locations (uniform grid is defined along the vector (alpha,beta) 
            starting at a*x+b*z = 0 and ending at a*x+b*z = 2*pi
        Refer to printPhysical() method's doc-string for description of all other input arguments
        """
        assert (type(nLoc) is int), 'nLoc must be int'
        a = self.flowDict['alpha']; b = self.flowDict['beta']; omega = self.flowDict['omega']
        gama = np.sqrt(a*a+b*b)
        K = self.flowDict['K']; L = self.flowDict['L']; M=-np.abs(self.flowDict['M'])
        N = np.int(self.N*interY)
        if etaLoc is None:  etaLoc = np.arange(0., 4.*np.pi/gama, 2.*np.pi/gama/nLoc)
        else: 
            assert isinstance(etaLoc,np.ndarray), 'etaLoc must be a numpy array'
            etaLoc = etaLoc.reshape(etaLoc.size)
        if (a == 0.) and (b == 0.):
            warn('Both alpha and beta are zero for the flowField. Printing a field at (x,z)=(0,0)')
            nLoc=1; etaLoc = np.zeros(1)
        
        if tLoc is None:
            if omega != 0.: tLoc = np.arange(0,2.*np.pi/omega, 2.*np.pi/b/7.)
            else: tLoc = np.zeros(1)
        if yLoc is None:
            yLoc = chebdif(interY*N,1)[0]
            yLocFlag = False
        else:
            yLocFlag = True
            assert isinstance(yLoc,np.ndarray) and (yLoc.ndim == 1), 'yLoc must be a 1D numpy array'
            
        assert type(yOff) is np.float or (type(yOff) is np.float64), 'yOff characterizes surface deformation and must be of type float'
        assert isinstance(fName,str), 'fName must be a string'
        if '.dat' in fName[-4:]: fName = fName[:-4]
        
        assert self.nd == 3, 'printPhysicalPlanar() is currently written to handle only 3C velocity fields'
        assert isinstance(etaLoc,np.ndarray) and isinstance(tLoc,np.ndarray),\
            'xLoc, zLoc, and tLoc must be numpy arrays'
        
        
        if pField is None: pField = self.solvePressure(divFree=False,nonLinear=True)[0]
        else:
            assert pField.size == self.size//3, 'pField must be the same size of each component of self'
        obj = self.appendField(pField)   # Appending pField to velocity field
        
        dataArr = np.zeros((8,tLoc.size,etaLoc.size,yLoc.size))
        
        if interY != 1 and not yLocFlag:
            obj = obj.slice(N=yLoc.size)
        
        
        # .getField() requires x and z locations. So, mapping required etaLoc to xLoc (with zLoc=0), if a != 0
        #       and to zLoc (with xLoc= 0) if a = 0. 
        # gama*eta = a*x + b*z;     If z = 0., x = gama*eta/a.   If x = 0., z = gama*eta/b
        if a != 0.: 
            xLoc = gama*etaLoc/a; zLoc = 0.
        else: 
            xLoc = 0. ; zLoc = gama*etaLoc/b
        
        if yLocFlag: field = obj.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc, yLoc=yLoc)
        else: field = obj.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc)
        # The output of .getField(), field, will be of shape (tLoc.size, xLoc.size, zLoc.size, obj.nd, yLoc.size)
        #     But either xLoc.size or zLoc.size is 1, compressing that axis:
        field = field.reshape((tLoc.size, etaLoc.size, obj.nd, yLoc.size))
        
        tLoc = tLoc.reshape(tLoc.size,1,1)
        etaLoc = etaLoc.reshape(1,etaLoc.size,1)
        
        # Assigning grid points:
        dataArr[1] = etaLoc; 
        dataArr[0] = yLoc + yOff*np.cos(gama*etaLoc-omega*tLoc)
        
        # In field, the scalars u,v,w,p are assigned on axis 2. dataArr needs these on axis 0, since the ascii is printed in columns
        for scal in range(4):
            dataArr[2+scal] = field[:,:,scal]
        
        # U_parallel and U_cross:
        dataArr[6] = a/gama*dataArr[2] + b/gama*dataArr[4]
        dataArr[7] = -b/gama*dataArr[2]+ a/gama*dataArr[4]
        
        if 'eps' in self.flowDict: eps = self.flowDict['eps']; g = eps*a
        else: eps = 1.0E-9; g = 0
        if a != 0.: theta = int(np.arctan(b/a)*180./np.pi)
        else: theta = 90
        variables = 'VARIABLES = "Y", "eta", "U", "V", "W", "P", "U_pl", "U_cr" \n'
        zoneName = 'T'+str(theta)+'E'+str(-np.log10(eps))+'G'+str(g)+'Re'+str(self.flowDict['Re'])
        zone = 'ZONE T="'+zoneName+ '", I='+str(yLoc.size)+', J='+str(etaLoc.size)+', DATAPACKING=POINT'
        if tLoc.size == 1:
            #np.savetxt(fName+'.csv', dataArr.reshape((7,dataArr.size//7)).T,delimiter=',')
            #tempArr = dataArr.reshape(dataArr.size)
            title = 'TITLE= "Flow (planar) in wavy walled channel with a='+str(a)+', b='+str(b)+\
                ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
            hdr = title+variables+zone
            np.savetxt(fName+'.dat',dataArr.reshape(8,dataArr.size//8).T, header=hdr,comments='')
            print('Printed physical field to file %s.dat'%fName)
        else:
            for tn in range(tLoc.size):
                title = 'TITLE= "Flow (planar) in wavy walled channel at t='+str(tLoc[tn,0,0])+' with a='+str(a)+', b='+str(b)+\
                    ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
                hdr = title+variables+zone
                np.savetxt(fName+str(tn)+'.dat', dataArr[:,tn].reshape((8,dataArr[:,tn].size//8)).T,header=hdr,comments='')
            print('Printed %d time-resolved physical fields to files %sX.dat'%(tLoc.size,fName))
        
        return

    def zero(self):
        """Returns an object of the same class and shape as self, but with zeros as entries"""
        obj = self.copy()
        obj[:] = 0.
        return obj


    def identity(self):
        return self
    
