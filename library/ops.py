import numpy as np
import scipy as sp
import pseudo 
import glob
from warnings import warn
import blasius
import sys
import minimize
import os
import h5py
from scipy.integrate import quad
assert sys.version_info >=(3,5), "The infix operator used for matrix multiplication isn't supported in versions earlier than python3.5. Install 3.5 or fix this code."

""" 
ops.py
Defines linear operators, such as OSS, resolvent, resolvent with eddy, etc... 
Problem definition must be supplied with keyword arguments

IMPORTANT: Everything here is written only for internal nodes. 
            Variables at the wall are always ignored. 
            y,v represent wall-normal
"""
covDataDir = os.environ['DATA']+ 'cov/'
class linearize(object):
    def __init__(self, N=50, U=None,dU=None, d2U=None,flowClass="channel",Re=10000., **kwargs):
        """
        Initialize OSS class instance. 
        Inputs:
            N(=50)      :   Number of internal nodes (x2 for BL to cover eta <0)
            U(=None)    :   Spatiotemporal mean streamwise velocity
            dU(=None)   :   dU/dy
            d2U(=None)  :   d2U/dy2
            flowClass(="channel")   : "channel", "couette", or "bl"

        Attributes: 
            N (no. of nodes), y (node location) 
            D1, D2, D4: Differentiation matrices
            w:  Weight matrix for defining integral norms. Use as np.dot(w,someFunctionOn_y)
            U, dU, d2U, flowClass:  Mean velocity and its derivatives, and flowClass
            __version__ :   Irrelevant to code, keeps track of development.
            Re: Reynolds number. Can be either 
        flowClass determines the operators D1, D2 and D4 to be used, and also U and d2U if not supplied.
        If U and d2U are not supplied, initialize with laminar velocity profile

        Methods:
            _weightVec, _weightMat: Returns clencurt-weighted versions of 1-d and 2-d arrays, 
                                    so that 2-norm of the weighted versions give energy norm of the original.
            _deweightVec, _deweightMat: Revert weighted versions to original unweighted versions 
            OS, OSS:    Orr-Sommerfeld and Orr-Sommerfeld-Squire matrix operators 
                            (coefficient matrix multiplying df/dt is included in the returned matrix)
            eig, svd:   Calls numpy.linalg's eig and svd routines, with kwarg for weighting.
            resolvent:  Currently unavailable.

            
        """
        N = np.int(N)
        if flowClass == "bl":
            Y = kwargs.get('Y',15.)
            if (N<70) or (Y<10.):
                warn("BLs need lots of nodes.. N and Y should be set at >~ 100 and >~ 15, currently they are %d, %.1g"%(N,Y))
            y,DM = pseudo.chebdifBL(N,Y=Y)    # BL code is written to return D only on internal nodes
            D1  = DM[:,:,0]
            D2 = DM[:,:,1]
            D4 = pseudo.cheb4cBL(N, Y=Y)
            w = pseudo.clencurtBL(N)  
        else:
            if flowClass not in ("channel" , "couette"):
                print("flowClass is not set to 'channel', 'bl', or 'couette'. Defaulting to 'channel'.....")
                flowClass = "channel"
            y,DM = pseudo.chebdif(N,2)
            # For now, channel code is written to include nodes at the wall. 
            D1 = DM[:,:,0]
            D2 = DM[:,:,1]
            D4 = pseudo.cheb4c(N)       # Imposes clamped BCs
            w = pseudo.clencurt(N)      # Weights matrix
       
        self.__version__ = '6.0.1'    # m.c, m is month and c is major commit in month
        self.N  = N
        self.y  = y
        self.D1 = D1; self.D = D1
        self.D2 = D2
        self.D4 = D4
        self.w  = w
        self.Re = np.float(Re)

        if (U is None):
            if flowClass == "channel":
                U = 1. - self.y**2; dU = -2.*self.y; d2U = -2.*np.ones(U.size)
            elif flowClass == "couette":
                U = self.y; dU = np.ones(U.size); d2U = np.zeros(U.size)
            elif flowClass == "bl":
                U,dU = blasius.blasius(self.y)
                d2U = np.dot( self.D2, (U-1.) )

            else:
                print("flowClass isn't any of the 3 accepted ones... Weird. Setting to channel....")
                flowClass = "channel"
        elif U.size != self.N: 
            U = U[1:1+self.N]

        if dU is None: 
            dU = np.dot(self.D1, U-1.)  # U-1 is zero as y goes to \inf for BLs
            # For channel and couette, subtracting 1 doesn't make any difference. 
        elif dU.size != self.N: 
            dU = dU[1: 1+self.N]

        if d2U is None:
            d2U = np.dot(self.D2, U-1.)
        elif d2U.size != self.N: 
            d2U = d2U[1: 1+self.N]

        self.U  = U
        self.dU = dU
        self.d2U = d2U
        self.flowClass = flowClass

        print("Initialized instance of 'linearize', version %s." %(self.__version__),
                "New in this version: pseudo module is defined for internal nodes ",
                "To fix: Eddy viscosity, resolvent, and svd are currently not supported.",sep="\n")

        return
        
       
    def _weightVec(self, vec):
        """Pre-multiply vectors with the square-root of weights, 
            so that a 2-norm  of the weighted vector gives the energy norm of the non-weighted vector 
        Call as self._weightVec(vec)"""
        w = self.w
        q = np.sqrt(w).reshape((1,self.N))  # Use sqrt of weights so that < qv, (qv).H > = energy norm of v
        shape0 = vec.shape

        weightedVec = vec.reshape((vec.size//self.N, self.N)) * q

        return weightedVec.reshape(shape0)

    
    def _weightMat(self, mat):
        """Pre-multiply matrices (2d-array) with the square-root of weights, and post-multiply with inv of sqrt of weights
            so that optimal gains are calculated over the energy norm without having to supply weight matrices 
        Call as self._weightMat(mat)"""
        w = self.w
        assert w.ndim == 1
        Q  = np.diag(np.sqrt(self.w))
        Qi = np.diag(1./np.sqrt(self.w))
        N = self.N

        for n1 in range( mat.shape[0]//N):
            for n2 in range(mat.shape[1]//N):
                mat[ n1*N: (n1+1)*N, n2*N: (n2+1)*N] = np.dot( Q, np.dot(mat[ n1*N: (n1+1)*N, n2*N: (n2+1)*N], Qi))

        return mat

    def _deweightVec(self,vec):
        """Pre-multiply vectors with the inverse of square-root of weights, 
            to obtain the physical vector from weighted vectors
        Call as self._unweightVec(vec)"""
        w = self.w
        qi = (1./np.sqrt(w)).reshape((1,self.N))  # Use sqrt of weights so that < qv, (qv).H > = energy norm of v
        shape0 = vec.shape

        deweightedVec = vec.reshape((vec.size//self.N, self.N)) * qi

        return deweightedVec.reshape(shape0)

    def _deweightMat(self,mat):
        """Pre-multiply matrices (2d-array) with the inv of square-root of weights, and post-multiply with sqrt of weights
            to recover the operator matrices from weighted ones .
        Call as self._weightMat(mat)"""
        w = self.w
        assert w.ndim == 1
        Q  = np.diag(np.sqrt(self.w))
        Qi = np.diag(1./np.sqrt(self.w))
        N = self.N

        for n1 in range( mat.shape[0]//N):
            for n2 in range(mat.shape[1]//N):
                mat[ n1*N: (n1+1)*N, n2*N: (n2+1)*N] = np.dot( Qi, np.dot(mat[ n1*N: (n1+1)*N, n2*N: (n2+1)*N], Q))

        return mat

    def OSS(self, **kwargs):
        """ Define the Orr-Sommerfeld Squire operator matrix. 
        Inputs:
            kwargs: Dictionary containing wavenumbers 
                        Defaults are (a,b) = (2.5, 20./3.)
        Outputs:
            OSS matrix
        What this OSS matrix is is given in function comments.
        Essentially, I define OSS*[vel,vor].T =: i.omega*[vel,vor].T
            """
        # For consistency, I use 2d numpy arrays instead of numpy matrices
        # Where dot products are needed, use np.dot(A,B) instead of A*B


        a = kwargs.get('a', 2.5)
        b = kwargs.get('b', 20./3.)
        Re= self.Re
        k2 = a**2 + b**2

        # We build the matrix only for internal nodes, so use N-2
        I = np.identity(self.N )
        Z = np.zeros(self.D1.shape )

        # All attributes of self ignore walls (in variables and differentiation matrices)
        D1_ = self.D1
        D2_ = self.D2
        D4_ = self.D4    
        U_ = np.diag(self.U)
        dU_ = np.diag(self.dU)
        d2U_ = np.diag(self.d2U)
        # U and d2U have to be diagonal matrices to multiply other matrices
     
        # The linearized equation looks like this (from Schmid & Henningson, 2001)
        # -i.omega  [vel]  =  [DeltaInv* LOS ,  Z  ] [vel]  
        #           [vor]  =  [-i.b.dU       ,  LSQ] [vor]  

        # where LOS =   1/Re . (D2-k^2 I)^2 + i.a.d2U - i.a.U.(D2-k^2 I)
        #       LSQ =   1/Re . (D2-k^2 I) - i.a.U   

        LOS  = (1./Re) * (k2*k2*I - 2.*k2*D2_ + D4_ ) + 1.j*a* d2U_ - 1.j*a *( U_ @ ( D2_ - k2 *I ) )   
        DeltaMat = D2_ - k2*I        
        DeltaInv = np.linalg.solve(DeltaMat, I) 

        LSQ  = (1./Re)* (D2_ -k2*I ) - 1.j*a*U_   

        OSS =  np.vstack(( np.hstack( (DeltaInv @ LOS, Z   ) ),\
                           np.hstack( (-1.j*b*dU_    ,  LSQ) ) ))

        return OSS



    def dynamicsMat(self,**kwargs):
        """ The dynamics matrix relating velocity-vorticity to their time-derivatives:
            [w_t, eta_t] = A [w,eta].
            This dynamics matrix A is the same as the OSS matrix
        """
        return self.OSS(**kwargs)

    def velVor2primitivesMat(self,**kwargs):
        """ Defines a matrix convertMat: [u,v,w]^T = convertMat @ [w,eta]^T.
        v is spanwise, w is wall-normal. 
        Input: 
            dict with wavenumbers (a,b) = (1,0) 
        Output:
            3N x 2N matrix convertMat """
        warn("z,w represent wall-normal, and y,v represent spanwise.")

        a = kwargs.get('a', 2./3. )
        b = kwargs.get('b', 20./3.)
        
        Z = np.zeros((self.N, self.N), dtype=np.complex)
        I = np.identity(self.N, dtype=np.complex)

        convertMat = np.vstack((
            np.hstack(( 1.j*a*self.D, -1.j*b*I )),
            np.hstack(( (a**2+b**2)*I,    Z    )),
            np.hstack(( 1.j*b*self.D,  1.j*a*I )) ))

        convertMat = convertMat/(a**2 + b**2)

        return convertMat
        

    def resolvent(self,**kwargs):
        pass
    
    def eig(self, mat, b=None, weighted=True, **kwargs):

        if weighted:
            mat = self._weightMat(mat)

        if b is None:
            evals, evecs = sp.linalg.eig(mat)
            b = np.identity(mat.shape[0])
        else:
            if weighted:
                b = self._weightMat(b)
            evals, evecs = sp.linalg.eig(mat, b=b)

        eignorm = np.linalg.norm(  np.dot(mat, evecs) - np.dot(np.dot(b,evecs),  np.diag(evals)) )
        print("Eigenvalue solution returned with error norm:",eignorm)
        if weighted:
            evecs = self._deweightVec(evecs)

        return evals, evecs
        

    def svd(self, mat, weighted=True): 
        pass

   

class statComp(linearize):
    def __init__(self,a=0.25,b=2./3.,**kwargs):
        """
        Initialize a case for covariance completion. 
        Inherits class 'linearize'. See this class for input arguments.
        The extra inputs are the streamwise and spanwise wavenumbers, a,b.
        Attributes: 
            N (no. of nodes), y (node location) 
            D1, D2, D4: Differentiation matrices
            w:  Weight matrix for defining integral norms. Use as np.dot(w,someFunctionOn_y)
            U, dU, d2U, flowClass:  Mean velocity and its derivatives, and flowClass
            __version__ :   Irrelevant to code, keeps track of development.
            a,b,Re: wavenumbers and Reynolds number. 
            covMat: Covariance matrix from DNS
            
        flowClass determines the operators D1, D2 and D4 to be used, and also U and d2U if not supplied.
        If U and d2U are not supplied, initialize with laminar velocity profile

        Inputs:
            All the keyword args as linearize, and
            a (float=2.): Streamwise wavenumber
            b (float=4.): Spanwise wavenumber
            In kwargs:
                covMat: Covariance matrix from DNS 
                structMat: Matrix that decides which entries of covMat are used as constraint

        Methods:
            dynamicsMat (weighted negative of the OSS matrix)
            outputMat   (weighted matrix for converting velocity vorticity to velocities)
        """
        kwargs['Re'] = kwargs.get('Re',186.)    # If ReTau is not supplied, use 186 
        kwargs['N'] = kwargs.get('N', 48)
        if (kwargs.get('U',None) is None) and (kwargs.get('flowClass','channel') is 'channel'):
            outDict = turbMeanChannel(**kwargs)
            kwargs.update(outDict)

        super().__init__(**kwargs)  # Initialize linearize subinstance using supplied kwargs
        self.a = a
        self.b = b
        if 'covMat' not in kwargs:
            warn("Need to rewrite covMat bit to allow ReTau=186 mats")
            covfName = glob.glob(covDataDir+'cov*l%02dm%02d.npy'%(a/a0,b/b0))[0]
            covMat = np.load(covfName)
            print("covMat was not supplied. Loaded matrix from %s ..."%(covfName))
        else:
            covMat = kwargs['covMat']
        assert covMat.shape == (3*self.N, 3*self.N)
        self.covMat = covMat

        if ('structMat' not in kwargs) or (not isinstance(kwargs['structMat'], np.ndarray)):
            print("structMat has not been supplied or is not a numpy array. Using one-point covariances for uu, vv, and ww, as well as the Reynolds shear stresses.")
            structMat = np.identity(3*self.N)   # All one-point normal stresses
            N = self.N
            structMat[range(N,3*N), range(2*N)] = 1     # One-point vu and wv
            structMat[range(2*N,3*N), range(N)] = 1     # One-point wu
            structMat[range(2*N), range(N,3*N)] = 1     # One-point uv and vw
            structMat[range(N), range(2*N,3*N)] = 1     # One-point uw
            #structMat[range(2*N,3*N), range(2*N,3*N)] = 0   # Ignoring ww one-point
            self.structMat = structMat
        else:
            self.structMat = kwargs['structMat']


        return

    def dynamicsMat(self):
        """ The dynamics matrix describing evolution of [w,eta]^T at (self.a, self.b).
        It is the negative of the weighted OSS operator.
        """
        return -self._weightMat( self.OSS(a=self.a, b=self.b) )

    def outputMat(self):
        """ The output matrix to get [Qu,Qv,Qw] from [Qw,Q eta] at (self.a, self.b).:w
        Q is the weight matrix so that ||(Qu)* Qu||_2 gives the energy of u.
        """
        unWeightedOutMat = self.velVor2primitivesMat(a=self.a, b=self.b)
        return self._weightMat(unWeightedOutMat)


    def completeStats(self,savePrefix='outStats',**kwargs):
        """ Completed statistics. 
        Inputs:
            savePrefix (='outStats'):   If not None, save output of stat completion (see below), along with self.a, self.b, self.Re, self.N as attributes in a hdf5 file.
            kwargs:     Optional keyword arguments. Can include key 'savePath' 
        Outputs:
            Output is a dict with keys:
            X: Completed covariance matrix
            Z: RHS of the Lyapunov equation
            Y1, Y2: Show up in the AMA algorithm. Not important.
            flag: Convergence flag
            steps: Number of steps at exit of AMA
            funPrimalArr, funDualArr: evaluations of the primal and dual residual functions at each step
            dualGapArr: Expected difference between primal and dual formulation."""
        kwargs['rankPar'] = kwargs.get('rankPar',200.)
        print("Parameters of the statComp instance are:")
        print("a:%.2g, b:%.2g, Re:%d, N:%d, rankPar:%.1g"%(self.a, self.b, self.Re, self.N, kwargs['rankPar']))
        statsOut = minimize.minimize( self.dynamicsMat(),
                outMat = self.outputMat(), structMat = self.structMat,
                covMat = self.covMat, **kwargs)
        if savePrefix is not None:
            fName = kwargs.get('savePath','') + savePrefix
            fName = fName + 'R%dN%da%02db%02d.hdf5'%(self.Re, self.N, self.a, self.b)
            try:
                with h5py.File(fName,"w") as outFile:
                    outStats = outFile.create_dataset("outStats",data=statsOut['X'],compression='gzip')

                    for key in self.__dict__.keys():
                        # Saving all attributes of the class instance for later regeneration
                        if isinstance(self.__dict__[key], np.ndarray):
                            outFile.create_dataset(key, data=self.__dict__[key],compression='gzip')
                        else:
                            outStats.attrs[key] = self.__dict__[key]
                    for key in statsOut.keys():
                        # Saving output statistics
                        if isinstance(statsOut[key], np.ndarray):
                            outFile.create_dataset(key, data=statsOut[key],compression='gzip')
                        else:
                            outStats.attrs[key] = statsOut[key]

                print("saved statistics to ",fName)
            except:
                print("Could not save output stats for whatever reason..")

        return statsOut


    def decomposeZ(self, Z=None, **kwargs):
        """ Refer to the non-class function decompose Z in this module"""
        if not hasattr(self,Z): self.Z = Z
        self.B, self.H, self.S = decomposeZ(self.Z,**kwargs)
        return





#===========================================================================================
#===========================================================================================
#===========================================================================================
# Non-class functions
def decomposeZ(Z, **kwargs):
    """ Decompose Z = BH* + HB*
    Following Qdecomposition of Zare and Jovanovic
    Inputs:
        Z:  The Lyapunov matrix
            Can also be an attribute of self
    Outputs:
        None. Populates the following attributes of self:
            B:  Input matrix
            H:  Forcing covariance
            S:  (= BH*)
    """
    Z = np.asarray(Z)   # Use np.ndarray instead of np.matrix, for consistency
    # Z is supposed to be a normal matrix. But just to get rid of any errors,
    Z = (Z + Z.conj().T)/2.
    n,m = Z.shape

    normZ = np.linalg.norm(Z, ord=2)    # Max singular value
    threshold = kwargs.get('threshold',1.0e-12) * normZ
    # eigvals with abs less than threshold will be considered to be zeros

    # Eigenvalue decomposition of Z
    evals, evecs = np.linalg.eig(Z)
    # Note: evals is a 1-d array, unlike MATLAB's diagonal matrix
    # evals are all real because Z is a normal matrix

    # Signature of Z, i.e. number of positive (piZ), negative (nuZ), and zero (deltaZ) eigenvalues
    piZ = np.sum( ( evals > threshold ).astype(np.int))
    nuZ = np.sum( ( evals < -threshold).astype(np.int))
    deltaZ  = n - piZ - nuZ


    # Set diagonal entries to 1, -1, or 0 according to the signature, call this D1
    #       See Zare 2016, Low complexity modelling ......
    evals1pi = np.diag( (evals > threshold).astype(np.int) )
    evals1nu =-np.diag( (evals <-threshold).astype(np.int) )
    evals1all   = evals1pi + evals1nu  # Positive evals set to 1, negatives to -1, zeros stay 0

    # Weight the eigenvectors by sqrt(|evals|) to account for the above change
    evalspi = np.diag(evals) * evals1pi     # Diagonal matrix with only evals>0
    evalsnu = np.diag(evals) * evals1nu     # Diagmat with only evals < 0
    evals_tmp = evalspi + evalsnu + ( np.identity(n) - np.abs(evals1all) )
        # abs(evals) whenever evals != 0, 1 when evals=0
    evecsW = evecs @ np.sqrt(evals_tmp)  # evecs weighted by sqrt(|evals|) for evals != 0


    # Sort evals, and reorder evecs accordingly
    piZind = np.nonzero( np.diag(evals1pi) )[0]    # Indices for evals > 0
    nuZind = np.nonzero( np.diag(evals1nu) )[0]    # Indices for evals < 0
    deltaZind = np.nonzero(  np.identity(n) - np.abs(evals1all) )[0]   # for evals = 0

    reorderInd = np.concatenate( (piZind, nuZind, deltaZind) )

    evals1Sorted = np.diag( np.concatenate((
                        np.ones(piZ), -np.ones(nuZ), np.zeros(deltaZ) )) )
    # Eigenvalues (1s, -1s, and 0s) sorted 

    evecsSorted = evecsW[:,reorderInd]
    # Weighted evecs sorted according to evals


    # Compute B, H, and S (refer to Zare 2016)
    evals1Sorted = 2.* evals1Sorted
    evecsSorted = np.sqrt(0.5) * evecsSorted

    if piZ <= nuZ:
        diffZ = nuZ - piZ
        Bhat = np.vstack(( 
            np.hstack(( np.identity(piZ),   np.zeros((piZ, diffZ))  )),
            np.hstack(( np.identity(piZ),   np.zeros((piZ, diffZ))  )),
            np.hstack(( np.zeros((diffZ,piZ)),np.identity(diffZ)    )),
            np.zeros(( deltaZ, nuZ )) 
            ))
        Hhat = np.vstack(( 
            np.hstack(( np.identity(piZ),   np.zeros((piZ, diffZ))  )),
            np.hstack((-np.identity(piZ),   np.zeros((piZ, diffZ))  )),
            np.hstack(( np.zeros((diffZ,piZ)),-np.identity(diffZ)   )),
            np.zeros(( deltaZ, nuZ )) 
            ))
    else:
        diffZ = piZ - nuZ
        Bhat = np.vstack((
            np.hstack(( np.identity(diffZ),     np.zeros(( diffZ, nuZ ))    )),
            np.hstack(( np.zeros((nuZ,diffZ)),  np.identity(nuZ)            )),
            np.hstack(( np.zeros((nuZ,diffZ)),  np.identity(nuZ)            )),
            np.zeros((deltaZ,piZ))
            ))
        Hhat = np.vstack((
            np.hstack(( np.identity(diffZ),     np.zeros(( diffZ, nuZ ))    )),
            np.hstack(( np.zeros((nuZ,diffZ)),  np.identity(nuZ)            )),
            np.hstack(( np.zeros((nuZ,diffZ)), -np.identity(nuZ)            )),
            np.zeros((deltaZ,piZ))
            ))
    
    B = evecsSorted @ Bhat
    H = evecsSorted @ Hhat
    S = B @ H.conj().T

    return B,H,S



def loadStatComp(fName):
    """ Create statComp instance from an earlier run saved/dumped as a hdf5 file
    Inputs:
        fName
    Outputs:
        statComp instance"""
    with h5py.File(fName,"r") as inFile:
        outStats = inFile['outStats']
        # All floats such as a,b,Re,.. are attributes of outStats
        # All arrays and matrices are datasets in inFile

        # Basic initialization of statComp instance:
        statInst = statComp(a=outStats.a, b=outStats.b, Re=outStats.Re, N=outStats.N, flowClass=outStats.flowClass)

        # Add mean velocity and its derivatives as attributes
        statInst.U = inFile['U']
        statInst.dU = inFile['dU']
        statInst.d2U = inFile['d2U']
    return

        
def turbMeanChannel(N=191,Re=186.,**kwargs):
    """
    Turbulent mean velocity profile, and its first two derivatives.
    Inputs:
        N (=191):   Number of internal Chebyshev nodes 
                        (coz my ReTau=186 simulation has 192 internal cells = 193 edges including the walls)
        Re (=186):  Friction Reynolds number
    Outputs:
        Dictionary containing
            U
            dU
            d2U:    Turbulent mean and its derivatives on Chebyshev grid with N nodes, normalized by friction velocity
            z:  Internal chebyshev nodes
    """
    if Re == 186.:
        alfa = 46.5
        kapa = 0.61
    else:
        alfa = 25.4
        kapa = 0.426

    nuT = lambda zt: -0.5 + 0.5*np.sqrt( 1.+
                (kapa*Re/3.* (2.*zt - zt**2) * (3. - 4.*zt + 2.*zt**2) *
                             (1. - np.exp( (np.abs(zt-1.)-1.)*Re/alfa )   )    )**2)

    intFun = lambda xi: Re * (1.-xi)/(1. + nuT(xi)) 
    zArr,DM = pseudo.chebdif(N,2)
    D1 = DM[:,:,0]    # Same for diff mats - only internal nodes
    D2 = DM[:,:,1]

    # I use z \in {-1,1}, but the nuT based integral equation for U is designed to work for
    #   z \in {0,2}. So....
    zArr= 1.+zArr   # Doesn't matter if the mapping is 1-z or 1+z, coz U is symmetric about the centerline

    U = np.zeros(zArr.size)

    for ind in range(zArr.size):
        U[ind] = quad( intFun, 0., zArr[ind])[0]

    dU = D1 @ U
    d2U = D2 @ U
    outDict = {'U':U, 'dU':dU, 'd2U':d2U,'z':zArr-1.}

    return outDict 






        

