import numpy as np
import scipy as sp
import pseudo 
from warnings import warn
import blasius
""" 
ops.py
Defines linear operators, such as OSS, resolvent, resolvent with eddy, etc... 
Problem definition must be supplied with keyword arguments

IMPORTANT: Everything here is written only for internal nodes. 
            Variables at the wall are always ignored. 
"""

class linearize(object):
    def __init__(self, N=50, U=None,dU=None, d2U=None,flowClass="channel", **kwargs):
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
            y,DM = pseudo.chebdif(N+2,2)
            # For now, channel code is written to include nodes at the wall. 
            y = y[1:-1]     # Ignore both walls
            DM = DM[1:-1, 1:-1]
            D1 = DM[:,:,0]
            D2 = DM[:,:,1]
            D4 = pseudo.cheb4c(N+2)       # Imposes clamped BCs
            w = pseudo.clencurt(N+2)      # Weights matrix
            w = w[1:-1]
       
        self.__version__ = '3.1'    # m.c, m is month and c is major commit in month
        self.N  = N
        self.y  = y
        self.D1 = D1
        self.D2 = D2
        self.D4 = D4
        self.w  = w

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
                "New in this version: diffmats and OSS validated for channel and BL LSA. ",
                "To fix: channel routines in pseudo.py to ignore wall-nodes, ",
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

    def OS(self, **kwargs):
        """ Define the Orr-Sommerfeld operator matrix (without the Squire equations). 
        Inputs:
            kwargs: Dictionary containing wavenumbers and Re
                        Defaults are (a,b,Re) = (1., 0., 10000.)
        Outputs:
            OS matrix
        What this OS matrix is is given in function comments.
        Essentially, I define OS*[vel].T =: i.omega*[vel].T
            """
        # For consistency, I use 2d numpy arrays instead of numpy matrices
        # Where dot products are needed, use np.dot(A,B) instead of A*B
        warn("This function only returns the matrix for Orr-Sommerfeld operator. For Squire, use self.OSS()")


        a = kwargs.get('a', 1.)
        b = kwargs.get('b', 0.)
        Re= kwargs.get('Re',10000.)
        k2 = a**2 + b**2

        # We build the matrix only for internal nodes, so use N-2
        I = np.identity(self.N )
        Z = np.identity(self.N )

        # All attributes of self ignore walls (in variables and differentiation matrices)
        D1_ = self.D1
        D2_ = self.D2
        D4_ = self.D4    
        U_ = np.diag(self.U)
        dU_ = np.diag(self.dU)
        d2U_ = np.diag(self.d2U)
        # U and d2U have to be diagonal matrices to multiply other matrices
     
        # The linearized equation looks like this (from Schmid & Henningson, 2001)
        # -i.omega [k^2 - D2,   Z]  [vel]  +  [LOS   ,  Z  ] [vel]  = [0]
        #          [Z,          I]  [vor]     [i.b.dU,  LSQ] [vor]  = [0]

        # where LOS = i.a.U.(k^2-D2) + i.a.d2U + 1/Re . (k^2 - D^2)^2
        #       LSQ = i.a.U  + 1/Re . (k^2 - D^2)
        LOS  = 1.j*a*np.dot(U_ , (k2*I - D2_ )) \
                + 1.j*a* d2U_  + 0.*2.j*a*dU_* D1_ \
                + (1./Re) * (k2*k2*I - 2.*k2*D2_ + D4_ )


        LHSmat =k2*I-D2_
        
        return np.dot(  np.linalg.inv(LHSmat), LOS)


    def OSS(self, **kwargs):
        """ Define the Orr-Sommerfeld Squire operator matrix. 
        Inputs:
            kwargs: Dictionary containing wavenumbers and Re
                        Defaults are (a,b,Re) = (1., 0., 10000.)
        Outputs:
            OSS matrix
        What this OSS matrix is is given in function comments.
        Essentially, I define OSS*[vel,vor].T =: i.omega*[vel,vor].T
            """
        # For consistency, I use 2d numpy arrays instead of numpy matrices
        # Where dot products are needed, use np.dot(A,B) instead of A*B


        a = kwargs.get('a', 1.)
        b = kwargs.get('b', 0.)
        Re= kwargs.get('Re',10000.)
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
        # -i.omega [k^2 - D2,   Z]  [vel]  +  [LOS   ,  Z  ] [vel]  = [0]
        #          [Z,          I]  [vor]     [i.b.dU,  LSQ] [vor]  = [0]

        # where LOS = i.a.U.(k^2-D2) + i.a.d2U + 1/Re . (k^2 - D^2)^2
        #               I think a + 2.i.a.dU is missing in the above expression
        #       LSQ = i.a.U  + 1/Re . (k^2 - D^2)
        LOS  = 1.j*a*np.dot(U_ , (k2*I - D2_ ))  + 1.j*a* d2U_ + (1./Re) * (k2*k2*I - 2.*k2*D2_ + D4_ )

        LSQ  = 1.j*a*U_  + (1./Re)* (k2*I - D2_ )
        #LSQ = -LSQ

        OSS0 =  np.vstack( ( np.hstack( (LOS      , Z  ) ),\
                             np.hstack( (1.j*b*dU_, LSQ) ) )  )

        LHSmat = np.vstack( ( np.hstack( (k2*I-D2_, Z) ),\
                              np.hstack( (Z       , I) ) ) )
        
        return np.dot(  np.linalg.inv(LHSmat), OSS0)

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
