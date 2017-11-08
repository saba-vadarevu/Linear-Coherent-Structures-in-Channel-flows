import numpy as np
import scipy as sp
from scipy.io import savemat
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds
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
covDataDir = os.environ['DATA']+ 'covN127/'
class linearize(object):
    def __init__(self, N=151, U=None,dU=None, d2U=None,flowClass="channel",Re=2000., **kwargs):
        """
        Initialize OSS class instance. 
        Inputs:
            N(=151)      :   Number of internal nodes (x2 for BL to cover eta <0)
            U(=None)    :   Spatiotemporal mean streamwise velocity
            dU(=None)   :   dU/dy
            d2U(=None)  :   d2U/dy2
            Re (=2000.) :   Reynolds number (friction based whenever U is normalized by friction velocity)
            flowClass(="channel")   : "channel", "couette", or "bl"
            kwargs['turb'] (=False): If True, initialized with turbulent base flow; Default is laminar.
                                    Turbulent flow can only be initialized for channel flow currently.

        Attributes: 
            N (no. of nodes), y (node location) 
            D1, D2, D4: Differentiation matrices
            w:  Weight matrix for defining integral norms. Use as np.dot(w,someFunctionOn_y)
            weightDict: Contains W1,W2,W3, etc.. for diagonal weight matrices
            U, dU, d2U, flowClass:  Mean velocity and its derivatives, and flowClass
            __version__ :   Irrelevant to code, keeps track of development.
            Re: Reynolds number. Can be either 
        flowClass determines the operators D1, D2 and D4 to be used, and also U and d2U if not supplied.
        flowState(="lam");  "lam" or  "turb", according to kwargs['turb']
        If U and d2U are not supplied, initialize with velocity profile according to flowState

        Methods:
            _weightVec, _weightMat: Returns clencurt-weighted versions of 1-d and 2-d arrays, 
                                    so that 2-norm of the weighted versions give energy norm of the original.
            _deweightVec, _deweightMat: Revert weighted versions to original unweighted versions 
            makeSystem: Returns OSS, input, and output matrices; optionally, adjoints of these matrices.
            dynamicsMat, inputMat, outputMat: Extract from makeSystem if others aren't needed
            
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
       
        self.__version__ = '10.1.1'    # m.c, m is month and c is major commit in month
        self.N  = N
        self.y  = y
        self.D1 = D1; self.D = D1
        self.D2 = D2
        self.D4 = D4
        self.w  = w
        self.Re = np.float(Re)
        
        flowState = 'lam'
        if (U is None):
            if flowClass == "channel":
                U = 1. - self.y**2; dU = -2.*self.y; d2U = -2.*np.ones(U.size)
                if kwargs.get('turb',False):
                    turbDict = turbMeanChannel(N=N,Re=Re)
                    U = turbDict['U']; dU = turbDict['dU']; d2U = turbDict['d2U']
                    flowState = 'turb'
            elif flowClass == "couette":
                U = self.y; dU = np.ones(U.size); d2U = np.zeros(U.size)
            elif flowClass == "bl":
                U,dU = blasius.blasius(self.y)
                d2U = np.dot( self.D2, (U-1.) )
                warn("Do not use self.weightDict or makeSystem('eddy'=True) ")

            else:
                print("flowClass isn't any of the 3 accepted ones... Weird. Setting to channel....")
                flowClass = "channel"
        elif U.size != self.N: 
            warn("Size of input U doesn't match N. Using values at nodes 1:1+N...")
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
        self.flowState = flowState
        self.weightDict = pseudo.weightMats(self.N)

        warn("Initialized instance of 'linearize', version %s.\n" %(self.__version__)+
                "New in this version: All system matrices are defined in makeSystem('eddy'=False,'adjoint'=False)\n"+
                "Eddy viscosity enhancement is allowed, but baseflow is reset to turbulent according to turbMeanChannel()\n"+
                "matNorm() defined to compute 2-norm and HS norm directly from unweighted matrices\n"+
                "weightDict is added as a class attribute")

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
        q = np.sqrt(w); q = np.tile(q, mat.shape[0]//q.size)
        Q  = np.diag(q)
        Qi = np.diag(1./q)

        return Q @ mat @ Qi

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

    def makeSystem(self, **kwargs):
        """ Define the Orr-Sommerfeld Squire operator matrix, and input and output matrices (B,C)
        Inputs:
            kwargs: Dictionary containing keys:
                a, b = (2.5, 20./3.); wavenumbers
                eddy = False;   If True, eddy viscosity is used (computed in turbMeanChannel, and U,dU,d2U reset accordingly)
                adjoint=False;  If True, returns adjoint matrices too
                
        Outputs:
            systemDict with keys
                A, B, C: OSS matrix, input matrix, and output matrix
                Aadj, Badj, Cadj: Adjoints of OSS, input and output matrices. These are only populated if input argument 'adjoint' is True

        What this OSS matrix is is given in function comments.
        Essentially, I define OSS*[vel,vor].T =: -i.omega*[vel,vor].T
        """


        a = kwargs.get('a', 2.5)
        b = kwargs.get('b', 20./3.)
        Re= self.Re
        k2 = a**2 + b**2
        #print("a, b, Re:",a,b,Re)

        # We build the matrix only for internal nodes, so use N-2
        I = np.identity(self.N ); I2 = np.identity(2*self.N); I1 = I
        Z = np.zeros(self.D1.shape ); Z1 = Z

        # All attributes of self ignore walls (in variables and differentiation matrices)
        D1_ = self.D1
        D2_ = self.D2
        D4_ = self.D4    
        U_ = np.diag(self.U)
        dU_ = np.diag(self.dU)
        d2U_ = np.diag(self.d2U)
        # U and d2U have to be diagonal matrices to multiply other matrices


        # Eddy viscosity; use only if kwargs['eddy'] is True
        if kwargs.get('eddy',False):
            # The problem I have is that I've set all variables on only the internal nodes,
            #   and they're all zero at the walls, except the viscosity. 
            # This isn't computationally efficient, but it's the easiest thing to do;
            # Compute the eddy viscosity afresh, and reset the velocities to match this.
            # Get y, nu, and U afresh, including entries at the wall
            turbDictTmp = turbMeanChannel(N=self.N,Re=self.Re,walls=True)
            DMwalls = pseudo.chebdif(self.N,2,walls=True)[1]   # diffmats including walls

            # Reset base flow
            warn('Base velocity is being reset using ops.turbMeanChannel...')
            self.flowState = 'turb'; self.flowClass = 'channel'
            self.U  = turbDictTmp['U'][1:-1]
            self.dU = self.D1 @ self.U
            self.d2U= self.D2 @ self.U

            # Get viscosity, including walls, and compute derivatives 
            nu   = turbDictTmp['nu']
            dnu  = DMwalls[:,:,0] @ nu
            d2nu = DMwalls[:,:,1] @ nu
            
            # Ignore values at the walls
            nu  = nu[1:-1]
            dnu = dnu[1:-1]
            d2nu= d2nu[1:-1]
        else:
            nu = np.ones(self.N)
            dnu = np.zeros(self.N)
            d2nu= np.zeros(self.N)
            

     
        # The linearized equation looks like this (from Schmid & Henningson, 2001)
        # -i.omega  [vel]  =  [DeltaInv* LOS ,  Z  ] [vel]  
        #           [vor]  =  [-i.b.dU       ,  LSQ] [vor]  

        # where LOS =   1/Re . (D2-k^2 I)^2 + i.a.d2U - i.a.U.(D2-k^2 I)
        #       LSQ =   1/Re . (D2-k^2 I) - i.a.U   
        # However, when eddy viscosity is included, these change to
        #       LOS =   i.a.d2U - i.a.U.(D2-k^2 I) + 1/Re*{ nu*(D2 - k^2 I)^2 + 2 nu' (D3 - k^2 D) + nu'' (D2 + k^2 I) }
        #       LSQ =   - i.a.U + 1/Re { nu (D2-k^2 I)  + nu' D}
        nuMat = np.diag(nu)
        dnuMat = np.diag(dnu)
        d2nuMat = np.diag(d2nu)

        # Laplacian and its inverse 
        Delta = D2_ - k2*I        
        DeltaInv = np.linalg.solve(Delta, I) 
        DeltaSq = (k2*k2*I - 2.*k2*D2_ + D4_ )
        D1 = D1_; D2 = D2_

        # Blocks of the OSS matrix
        LOS  = 1.j*a* d2U_ - 1.j*a *( U_ @ Delta ) + \
                (1./Re) *(  nuMat@ DeltaSq + 2.* dnuMat @ Delta @ D1_ + d2nuMat @ (D2_ + k2 * I  )    )
        LSQ  = - 1.j*a*U_  + (1./Re)*( nuMat @  Delta + dnuMat @ D1_ ) 
        Lco  = -1.j*b*dU_

        
        A =  np.vstack(( np.hstack( (DeltaInv @ LOS, Z   ) ),\
                           np.hstack( (Lco           ,  LSQ) ) ))
        
        B = np.vstack((
                        DeltaInv@ np.hstack(( -1.j*a*D1, -k2*I1, -1.j*b*D1 )),
                        np.hstack((1.j*b*I1, Z1, -1.j*a*I1))            ))
        
        C = (1./k2) * np.vstack((
                        np.hstack((1.j*a*D1, -1.j*b*I1)),
                        np.hstack((k2 * I1 , Z1       )),
                        np.hstack((1.j*b*D1,  1.j*a*I1)) ))
        systemDict = {'A':A, 'B':B, 'C':C, 'DeltaInv':DeltaInv}
        # Just coz I don't like having to do inversions too many times
        if kwargs.get('adjoint',False):
            if kwargs.get('eddy',False):
                if False:
                    # There's some discrepancy between Hwang's and Jovanovic's formulation
                    # Until I get to working on this bit, I'll use separate routines for the adjoints
                    LOSadj = 1.j*a*U_ @ Delta - 1.j*a*d2U_ + \
                            (1./Re)* (nuMat @ DeltaSq + 2.*dnuMat @ Delta @ D1 + d2nuMat @ (D2 + k2*I1) )
                    LSQadj = 1.j*a*U_ + (1./Re)*( nuMat @ Delta + dnuMat @ D1)
                    Lcoadj = 1.j*b*dU_

                    Aadj  = np.vstack((
                                        np.hstack(( DeltaInv @ LOSadj , Lcoadj )) , 
                                        np.hstack((       Z1          , LSQadj ))  ))
                    Badj = np.vstack(( 
                                        np.hstack(( -1.j*a*D1 @ DeltaInv,  -1.j*b*I1 )), 
                                        np.hstack(( -1.j*k2* DeltaInv   ,  Z1  )),
                                        np.hstack(( -1.j*b*D1 @ DeltaInv,  1.j *a*I1 ))   ))
                    Cadj = 1./k2 * np.vstack(( 
                                        np.hstack(( 1.j*a*D1,  k2*I1,  1.j*b*D1 )),
                                        np.hstack(( 1.j*b*I1,  Z1   ,  -1.j*a*I1))  ))
                elif False:
                    Badj = (1./k2) * np.vstack(( 
                                        np.hstack(( -1.j*a*D1  ,  -1.j*b*I1 )), 
                                        np.hstack(( k2 * I1    ,  Z1        )),
                                        np.hstack(( -1.j*b*D1  ,  1.j* a*I1 ))   ))
                    Cadj = (1./k2**2) * np.vstack(( 
                                        np.hstack(( 1.j*a*DeltaInv@D1, -k2*DeltaInv, 1.j*b*DeltaInv@D1 )),
                                        np.hstack(( 1.j*b*I1         , Z1          , -1.j*a*I1  ))   ))
                else:
                    # For now, just go with adjoints being complex conjugates
                    # This bit really needs fixing though
                    # Will get back to this later 
                    Aadj = A.conj().T
                    Badj = B.conj().T
                    Cadj = C.conj().T

            else:
                LOSadj = 1.j*a*U_  - 1.j*a* DeltaInv @ d2U_  + (1./Re)* DeltaInv  @ DeltaSq 
                LSQadj = 1.j*a*U_ + (1./Re)* Delta  
                Lcoadj = -1.j*b* DeltaInv @ dU_

                Aadj = np.vstack((
                                    np.hstack((  LOSadj , Lcoadj )),
                                    np.hstack((  Z1     , LSQadj ))    ))
                # Is this right? From definitions of JB05, doesn't look like it should be
                # But this is how he defines them.. So.. Check if this also follows from 
                #   the assumption of D^2 being Hermitian... 
                Badj = C.copy()
                Cadj = B.copy()

            systemDict.update({'Aadj':Aadj, 'Badj':Badj, 'Cadj':Cadj})

        return systemDict 



    def dynamicsMat(self,**kwargs):
        """ The dynamics matrix relating velocity-vorticity to their time-derivatives:
            [v_t, eta_t] = A [v,eta].
            This dynamics matrix A is the same as the OSS matrix
        """
        return self.makeSystem(**kwargs)['A']


    def velVor2primitivesMat(self,**kwargs):
        """ Defines a matrix convertMat: [u,v,w]^T = convertMat @ [v,eta]^T.
        Input: 
            dict with wavenumbers (a,b) = (1,0) 
        Output:
            3N x 2N matrix convertMat """
        return self.makeSystem(**kwargs)['C']

    def outputMat(self,**kwargs):
        return self.velVor2primitivesMat(**kwargs)

    def primitives2velVorMat(self, **kwargs):
        """ Defines a matrix convertMat: [v,eta]^T = convertMat @ [u,v,w]^T.
        Input: 
            dict with wavenumbers (a,b) = (2.5,20./3.) 
        Output:
            3N x 2N matrix convertMat """
        return self.makeSystem(**kwargs)['B']

    def matNorm(self, someMat,ord=2):
        """ Returns the H_infty norm (largest singular value) or Frobenius norm for 'someMat'
        The weighting of the matrix is done within the routine, so supply unweighted matrices 
        Inputs:
                someMat: Unweighted matrix to be normed
                ord =2:  Largest singular value for ord=2 or '2', and Frobenius norm for ord='HS' or 'fro' or 'Fro' """
        assert someMat.ndim ==2
        if someMat.shape[0] == self.N: W0 = self.weightDict['W1Sqrt']
        elif someMat.shape[0] == 2*self.N: W0 = self.weightDict['W2Sqrt']
        elif someMat.shape[0] == 3*self.N: W0 = self.weightDict['W3Sqrt']
        else: raise RuntimeError("Each dimension of the input matrix must be an integer multiple of self.N")
        if someMat.shape[1] == self.N: W1 = self.weightDict['W1SqrtInv']
        elif someMat.shape[1] == 2*self.N: W1 = self.weightDict['W2SqrtInv']
        elif someMat.shape[1] == 3*self.N: W1 = self.weightDict['W3SqrtInv']
        else: raise RuntimeError("Each dimension of the input matrix must be an integer multiple of self.N")

        H = W0 @ someMat @ W1
        if (ord == 2) or (ord == '2'):
            matrixNorm = svds(H,k=2)[1][0]
        elif (ord == 'HS') or (ord == 'fro') or (ord == 'Fro'):
            matrixNorm = np.linalg.norm(H, ord='fro')
        else: 
            matrixNorm = svds(H,k=2)[1][0]
            warn("Input argument 'ord' is not recognized. Returning largest singular value")

        return matrixNorm
    

   

class statComp(linearize):
    def __init__(self,a=0.25,b=20./3.,Re=186.,**kwargs):
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
        kwargs['N'] = kwargs.get('N', 62)
        if (kwargs.get('U',None) is None) and (kwargs.get('flowClass','channel') == 'channel'):
            # If U and its derivatives are not supplied, use curve-fit from ops.turbMeanChannel()
            outDict = turbMeanChannel(Re=Re,**kwargs)
            kwargs.update(outDict)

        warn("DO NOT USE THIS CLASS (STATCOMP). Use linearize instead.") 

        super().__init__(**kwargs)  # Initialize linearize subinstance using supplied kwargs
        self.a = a
        self.b = b
        self.Re = Re
        self.nu = kwargs.get('nu', np.ones(self.N))
        if 'covMat' not in kwargs:
            a0 = 0.25; b0 = 2./3.; N = self.N
            covfName = glob.glob(covDataDir+'cov*l%02dm%02d.npy'%(a/a0,b/b0))[0]
            print("covMat was not supplied. Loading matrix from %s ..."%(covfName))
            #covMatTemp = 0.5 * np.load(covfName)    
            # The 0.5 factor is needed because the weight matrix used when computing covarainces did not include this 0.5 
            covMatTemp = np.load(covfName)   # Don't need the 0.5 factor anymore, weighting's fixed.
            print("covMat from DNS data has y as spanwise. Reordering to have y as wall-normal......")
            covMat = covMatTemp.copy()
            covMat[0*N:1*N, 1*N:2*N ] = covMatTemp[0*N:1*N, 2*N:3*N]    # Assign uw* from DNS to uv* in linear modelling 
            covMat[0*N:1*N, 2*N:3*N ] = covMatTemp[0*N:1*N, 1*N:2*N]    # Assign uv* from DNS to uw* in linear modelling 

            covMat[1*N:2*N, 0*N:1*N ] = covMatTemp[2*N:3*N, 0*N:1*N]    # Assign wu* from DNS to vu* in linear modelling 
            covMat[1*N:2*N, 1*N:2*N ] = covMatTemp[2*N:3*N, 2*N:3*N]    # Assign ww* from DNS to vv* in linear modelling 
            covMat[1*N:2*N, 2*N:3*N ] = covMatTemp[2*N:3*N, 1*N:2*N]    # Assign wv* from DNS to vw* in linear modelling 
            
            covMat[2*N:3*N, 0*N:1*N ] = covMatTemp[1*N:2*N, 0*N:1*N]    # Assign vu* from DNS to wu* in linear modelling 
            covMat[2*N:3*N, 1*N:2*N ] = covMatTemp[1*N:2*N, 2*N:3*N]    # Assign vw* from DNS to wv* in linear modelling 
            covMat[2*N:3*N, 2*N:3*N ] = covMatTemp[1*N:2*N, 1*N:2*N]    # Assign vv* from DNS to ww* in linear modelling 
            print("Reordering complete... Remember to verify the ordering.")
            print("Remember that the covariance matrix is defined on clencurt-weighted velocity fields.")
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

    def makeTransformedSystem(self,weight=True,**kwargs):
        """ Returns the dynamics matrix A_psi describing evolution of psi, and output matrix C_psi relating psi to v
        Inputs:
            None 
        Outputs:
            A_psi:  Dynamics matrix; d_t psi = A_psi  psi  + forcing
            C_psi:  Output matrix;  W^1/2  v  =  C_psi  psi"""
           

        """ psi = Q^{1/2} phi, with Q = Q^{1/2}* Q^{1/2} and phi = [v, eta]^T  at (self.a, self.b).
            Q is defined such that the standard Euclidean norm < psi, psi > produces the energy norm,
                < W^1/2 v, W^1/2 v> = 1/2 \int_{-1}^{1}  ||u||^2 + ||v||^2 + ||w||^2  dy
            Comparing factors, we get Q = C* W C
            If A_phi describes the dynamics of phi, so that d_t phi = A_bar phi + forcing, 
            and A_psi describes the dynamics of psi, so that d_t psi = A psi + forcing, 
            then A_psi, the dynamics matrix, relates to A_phi (which is the OSS matrix) as
            A_psi = Q^{1/2} A_phi Q^{-1/2}
        """
        A_phi = self.dynamicsMat()  # OSS matrix describing the evolution of phi = [v, eta]^T
        C_phi = self.velVor2primitivesMat()    # v = C_phi * phi
        
        weightDict = pseudo.weightMats(self.N)
        Wsqrt = weightDict['W3Sqrt']
        W = weightDict['W3']
        
        Q = C_phi.conj().T  @ W @ C_phi
        Qsqrt = sqrtm(Q)     # Routine from scipy for square root of matrices
        QsqrtInv = np.linalg.solve( Qsqrt, np.identity(Qsqrt.shape[0]) )
        
        A_psi = Qsqrt @ A_phi @ QsqrtInv
        """ Matrix C_psi such that  W^1/2 v = C_psi psi
        The W^1/2 factor for v is there because we want the Euclidean norm of W^1/2 v to be the energy norm.
        From the definition of C_psi, and from psi = Q^1/2 phi (see above), v = C_phi phi, 
           C_psi = W^1/2  C_phi  Q^{-1/2} 
        """
        C_psi = Wsqrt @ C_phi @ QsqrtInv


        """ Matrix B_psi such that psi = B_psi  W1/2 v, 
        an inverse of C_psi in a sense.
        From psi = Q1/2 phi, phi = B_phi v, we can write
            psi = B_psi W1/2 v = (Q1/2 B_phi W-1/2) W1/2 v, 
            where B_phi comes from primitives2velVorMat()"""
        B_phi = linearize.primitives2velVorMat(self, a=self.a, b=self.b)
        wInv = 1./self.w
        WsqrtInv = np.diag( np.sqrt( np.concatenate(( wInv, wInv, wInv )) ))
        B_psi = Qsqrt @ B_phi @ WsqrtInv

        return A_psi, C_psi, B_psi

    def dynamicsMat(self,**kwargs):
        return linearize.dynamicsMat(self,a=self.a, b=self.b, **kwargs)

    def outputMat(self,**kwargs):
        return linearize.outputMat(self, a=self.a, b=self.b, **kwargs)
    
    def makeTransformedSystem(self,weight=True,**kwargs):
        """ Returns the dynamics matrix A_psi describing evolution of psi, and output matrix C_psi relating psi to v, based on JB05.
        Inputs:
            None 
        Outputs:
            A_psi:  Dynamics matrix; d_t psi = A_psi  psi  + forcing
            C_psi:  Output matrix;  W^1/2  v  =  C_psi  psi
            B_psi:  Complex conjugate of output matrix"""
           

        """         """
        a = self.a; b=self.b; N = self.N
        
        k2 = a**2 + b**2
        Z1 = np.zeros((N,N), dtype=np.complex)
        I1 = np.identity(1*N, dtype=np.complex)
        I2 = np.identity(2*N, dtype=np.complex)
        I3 = np.identity(3*N, dtype=np.complex)

        weightDict = self.weightDict
        W2  = weightDict['W2']
        W3  = weightDict['W3']
        W2s = weightDict['W2Sqrt']
        W2si= weightDict['W2SqrtInv']
        W3s = weightDict['W3Sqrt']
        W3si= weightDict['W3SqrtInv']

        Delta = self.D2 - k2 * I1
        Q = (1./k2) * np.vstack((   np.hstack(( -Delta, Z1)), 
                                        np.hstack(( Z1,  I1))    ))

        Q = (W2) @ Q
        Qs = sqrtm(Q)
        Qsi= np.linalg.solve(Qs, I2)


        Aphi = self.dynamicsMat()
        Cphi = self.velVor2primitivesMat()
        Bphi = self.primitives2velVorMat(custom=False)
        Apsi = Qs @ Aphi @ Qsi
        Bpsi = Qs @ Bphi @ W3si
        Cpsi = W3s@ Cphi @ Qsi
        
        return Apsi, Cpsi, Bpsi

    def stateTrans(self):
        """ Return Q^{1/2} in psi = Q^{1/2} phi """
        a = self.a; b=self.b; N = self.N
        
        k2 = a**2 + b**2
        Z1 = np.zeros((N,N), dtype=np.complex)
        I1 = np.identity(1*N, dtype=np.complex)
        I2 = np.identity(2*N, dtype=np.complex)
        I3 = np.identity(3*N, dtype=np.complex)

        weightDict = pseudo.weightMats(N=N)
        W2  = weightDict['W2']

        Delta = self.D2 - k2 * I1
        Q = (1./k2) * np.vstack((   np.hstack(( -Delta, Z1)), 
                                        np.hstack(( Z1,  I1))    ))

        Q = (W2) @ Q
        Qs = sqrtm(Q)
        return Qs

    
    def makeAdjSystem(self,**kwargs):
        """ Build adjoint system.
        Inputs:
            None
        Outputs: [Aadj, Cadj, Badj]
            Aadj: Adjoint of dynamics matrix
            Cadj: Adjoint of output matrix (C): W1/2 v = C psi
            Badj: Adjoint of inverse of output matrix (B): psi = B W1/2 v

            Here, W1/2 v is the clencurt-weighted velocity vector
                psi is the state, a weighted [v, eta]^T vector such that
                < psi, psi>_Euclidean = 0.5 \int_{-1}^1 ||u||^2 + ||v||^2 + ||w||^2 dy 
        """
        N = self.N; a = self.a; b = self.b; Re = self.Re
        k2 = a**2 + b**2
        I = np.identity(N); Z = np.zeros((N,N), dtype=np.complex)

        Umat = np.diag(self.U)
        dUmat = np.diag(self.dU)
        d2Umat = np.diag(self.d2U)


        A, C, B = self.makeSystem(**kwargs)
        Badj = C.copy()
        Cadj = B.copy()

        # Building Aadj:
        Aadj = np.zeros((2*N, 2*N), dtype=np.complex)
        Delta = self.D2 - k2 * I
        DeltaInv = np.linalg.solve(Delta, I)
        Aadj11 = 1.j*a*Umat - 1.j*a*DeltaInv @ d2Umat + \
                        1./Re*DeltaInv @ (Delta @ Delta)
        Aadj22 = 1.j*a*Umat + 1./Re*Delta
        Aadj21 = -1.j*b*DeltaInv @ dUmat

        Aadj = np.vstack(( 
                    np.hstack(( Aadj11, Aadj21 )),
                    np.hstack((  Z    , Aadj22 ))   ))
        # This Aadj here is just for the OSS matrix and does not account for the transformation in state
        # We need some additional matrices
        weightDict = pseudo.weightMats(self.N)
        Wsqrt = weightDict['W3Sqrt']
        W = weightDict['W3']
        C_phi = self.velVor2primitivesMat()
        
        Q = C_phi.conj().T  @ W @ C_phi
        Qsqrt = sqrtm(Q)     # Routine from scipy for square root of matrices
        QsqrtInv = np.linalg.solve( Qsqrt, np.identity(Qsqrt.shape[0]) )
        
        Aadj = Qsqrt @ Aadj @ QsqrtInv

        return Aadj, Cadj, Badj

    def covMatSymmetry(self):
        """ Impose symmetries on the covariance matrix, self.covMat 
        Even symmetry for uu, ww, uw, vv, and odd symmetry for uv, wv
        Inputs:
            None
        Outputs:
            None. Adds a new attribute covMatSymm to self"""
        symmIndArr = np.array([1,-1,1]).reshape(1,3) * np.array([1,-1,1]).reshape(3,1)
        covMatSymm = self.covMat.copy()
        N = self.N
        for ind0 in range(3):
            for ind1 in range(3):
                covBlock= covMatSymm[ind0*N:(ind0+1)*N , ind1*N: (ind1+1)*N ] 
                covBlock = 0.5*( covBlock + symmIndArr[ind0,ind1] * covBlock[::-1,::-1] )
                covMatSymm[ind0*N:(ind0+1)*N , ind1*N: (ind1+1)*N ] = covBlock
        self.covMatSymm = covMatSymm
        return


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
        A , C, B = self.makeSystem()
        Aadj, Cadj, Badj = self.makeAdjSystem()
        statsOut = minimize.minimize( A,
                outMat = C, structMat = self.structMat,
                covMat = self.covMat, outMatAdj = Cadj, dynMatAdj = Aadj, **kwargs)
        if savePrefix is not None:
            fName = kwargs.get('savePath','') + savePrefix
            a0 = 0.25; b0 = 2./3.
            fName = fName + 'R%dN%da%02db%02d.hdf5'%(self.Re, self.N, self.a//a0, self.b//b0)
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

    def saveSystem(self, fName='minSys.mat',prefix='../'):
        fName = prefix + fName
        A,C,B = self.makeSystem()
        if False:
            if not (fName.endswith('.hdf5')  or fName.endswith('.h5')):
                fName = fName.split('.')[0]
                fName = fName + '.hdf5'
            with h5py.File(fName,"w") as outFile:
                outFile.create_dataset('A', data=np.asmatrix(A))
                outFile.create_dataset('C', data=np.asmatrix(C))
                outFile.create_dataset('E', data=np.asmatrix(self.structMat))
                outFile.create_dataset('G', data=np.asmatrix(self.covMat))
            print("Successfully saved system matrices A (dynMat), C (outMat), E (structMat), and G (covMat) to file ", fName)
        else:
            if not (fName.endswith('.mat')):
                fName = fName.split('.')[0]
                fName = fName + '.mat'
            savemat(fName, {'A':A, 'C':C, 'E':self.structMat, 'G':self.covMat})
            print("Successfully saved system matrices A (dynMat), C (outMat), E (structMat), and G (covMat) to file ", fName)

        return




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
        walls (=False): If True, all returned entries are built to include the wall-nodes
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
    # However, if alfa and kapa are supplied as keyword arguments, use them instead
    alfa = kwargs.get('alfa', alfa)
    kapa = kwargs.get('kapa', kapa)
    kwargs['walls'] = kwargs.get('walls',False)
    #print("Using parameters Re=%.4g, alfa=%.4g, kapa=%.4g, N=%d"%(Re,alfa,kapa,N))

    nuT = lambda zt: -0.5 + 0.5*np.sqrt( 1.+
                (kapa*Re/3.* (2.*zt - zt**2) * (3. - 4.*zt + 2.*zt**2) *
                             (1. - np.exp( (np.abs(zt-1.)-1.)*Re/alfa )   )    )**2)

    intFun = lambda xi: Re * (1.-xi)/(1. + nuT(xi)) 
    zArr,DM = pseudo.chebdif(N,2,walls=kwargs['walls'])
    D1 = DM[:,:,0]    # Same for diff mats - only internal nodes
    D2 = DM[:,:,1]

    # I use z \in {-1,1}, but the nuT based integral equation for U is designed to work for
    #   z \in {0,2}. So....
    zArr= 1.-zArr   # Doesn't matter if the mapping is 1-z or 1+z, coz U is symmetric about the centerline

    U = np.zeros(zArr.size)
    U[0] = quad( intFun, 0., zArr[0])[0]

    for ind in range(1,zArr.size):
        U[ind] = U[ind-1] + quad( intFun, zArr[ind-1], zArr[ind])[0]

    dU = D1 @ U
    d2U = D2 @ U
    outDict = {'U':U, 'dU':dU, 'd2U':d2U,'y':zArr-1.,'nu': 1.+nuT(zArr)}

    return outDict 






        

