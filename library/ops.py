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
        Initialize OSS class instance
        Inputs:
            N   :   Number of internal nodes (x2 for BL to cover eta <0)
                        default: 35
            U   :   Spatiotemporal mean streamwise velocity
            d2U :   U''
            flowClass: "channel", "couette", or "bl"
                            default: channel
        flowClass determines the operators D2 and D4 to be used, and also U and d2U if not supplied.
        If U and d2U are not supplied, initialize with laminar velocity profile
        """
        if flowClass == "bl":
            warn("Code for boundary layers is not ready yet.. DO NOT TRUST THESE RESULTS.")
            y,DM = pseudo.chebdifBL(N+1,2)
            y = y[1:]   # Ignore the wall
            DM = DM[1:,1:]
            D  = DM[:,:,0]
            D2 = DM[:,:,1]
            D4 = np.dot(D2,D2)  # This has to change to employ clamped BCs
            w = pseudo.clencurt(N+1)  # This is very wrong. Need to use transformed weights
            w = w[1:]
        else:
            if flowClass not in ("channel" , "couette"):
                print("flowClass is not set to 'channel', 'bl', or 'couette'. Defaulting to 'channel'.....")
                flowClass = "channel"
            y,DM = pseudo.chebdif(N+2,2)
            y = y[1:-1]     # Ignore both walls
            DM = DM[1:-1, 1:-1]
            D1 = DM[:,:,0]
            D2 = DM[:,:,1]
            D4 = pseudo.cheb4c(N+2)       # Imposes clamped BCs
            w = pseudo.clencurt(N+2)      # Weights matrix
            w = w[1:-1]
        
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
                U = U[1:]; dU = dU[1:]
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
