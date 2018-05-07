"""
# pseudo.py
# Version 6.0.1
#
# IMPORTANT: In this version, all functions take as argument the number of internal nodes, denoted Nint
#           clencurt() modified to include a factor of 0.5, to normalize the integral which is from -1 to 1
# Contains (channel geometry) functions:
#   y,DM    = chebdif(Nint,M),     Differentiation matrices (Chebyshev collocation)
#   D4      = cheb4c(Nint),        Fourth differentiation matrix assuming clamped BCs (Chebyshev collocation)
#   w       = clencurt(Nint),      Clenshaw-Curtis weights
#   dotProd = chebdot(vec1,vec2, Nint),  dot product using Clenshaw-Curtis weights
#   norm2   = chebnorm(vec,Nint),  2-norm using Clenshaw-Curtis weights
#   norm2   = chebnorm2(vec,Nint), 2-norm using Clenshaw-Curtis weights
#   norm1   = chebnorm1(vec,Nint), 1-norm using Clenshaw-Curtis weights
#   coeffs  = chebcoeffs(f),    Chebyshev coefficients from collocation
#   coll    = chebcoll_vec(a),  Collocation values from Chebyshev coefficients
#   interp  = chebint(fk,x),    Interpolation from Chebyshev grid to a general grid
#   integral= chebintegrate(f), Integrate function values with BC f(y=-1) = 0
#   weightDict=weightMats(N),   clencurt weights arranged as different diagonal
# Similar functions for boundary layers (semi-infinite domain) using exponential mapping of nodes:
#   chebdifBL, cheb4cBL, clencurtBL, chebdotBL, chebnormBL, chebnorm1BL, chebnorm2BL, chebnormlBL
"""
## ----------------------------------------------------
## ACKNOWLEDGEMENT

# Implementation for chebdif and poldif is based on Weideman and Reddy's differentiation suite for Matlab.
# Most of the code is A line-by-line translation of their functions.
# This translation is done, in spite of available pseudo-spectral differentiation codes since the Matlab
# suite incorporates features that improve accuracy, which becomes important for calculating higher derivatives.
# Notably, the chebdif function uses trigonometric identities in [0,pi/2] to calculate x(k)-x(j), and higher 
# derivatives are explicitly calculated, instead of using D*D*..

# The webpage for the differentiation suite:
# http://dip.sun.ac.za/~weideman/research/differ.html

# 'clencurt' is from a blog by Dr. Greg von Winckel:
#	http://www.scientificpython.net/pyblog/clenshaw-curtis-quadrature	
# However, a minor change has been made- only the weight matrix is returned, as opposed to returning both the weights
#		and the nodes, as in the origion version on the blog


# Sabarish Vadarevu
# Mechanical engineering  
# University of Melbourne, Australia
# Email: Sabarish.Vadarevu@unimelb.edu.au 
# May, 2018
###--------------------------------------------------------

import numpy as np
from scipy.linalg import toeplitz
from scipy.fftpack import ifft
from warnings import warn

def chebdif(Nint,M,walls=False):
    """ y,DM =chebdif(Nint,M):  Differentiation matrices for Chebyshev collocation.
    Inputs: 
        Nint:  Number of internal nodes
        M:  Number of derivatives
    Outputs:
        y:  Chebyshev collocation nodes on (1,-1), 1d np.ndarray of size N
        DM: Differentiation matrices of shape (N,N,M).
                Extract m^th differentiation matrix as Dm = DM[:,:,m-1].reshape((N,N))
                For m^th derivative, invoke as 
                    f_m = np.dot(Dm, f), where f is a vector of function values on nodes 'y'
    """
    N = Nint + 2
    I = np.identity(N)		# Identity matrix
    #L = bool(I)

    n1 = np.int(np.floor(N/2.))		# Indices for flipping trick
    n2 = np.int(np.ceil(N/2.))

    # Theta vector as column vector
    k=np.vstack(np.linspace(0.,N-1.,N))		
    th=k*np.pi/(N-1.)

    x=np.sin( np.pi* np.vstack( np.linspace(N-1.,1.-N,N) ) /2./(N-1.) )	# Chebyshev nodes

    T = np.tile(th/2.,(1,N))
    DX= 2.*np.sin(T.T+T)*np.sin(T.T-T)	# Compute dx using trigonometric identity, improves accuracy

    DX= np.vstack((DX[0:n1,:], np.flipud(np.fliplr([-1.*el for el in DX[0:n2,:]]))))
    # DX = DX+I			# Replace 0s in diagonal by 1s (required for calculating 1/dx)
    np.fill_diagonal(DX, 1.)

    C= toeplitz(pow(-1,k))			# Matrix with entries c(k)/c(j)
    C[0,:] = [2.*el for el in C[0,:]]	
    C[N-1,:] = [2.*el for el in C[N-1,:]]
    C[:,0] = [el/2. for el in C[:,0]]
    C[:,N-1] = [el/2. for el in C[:,N-1]]


    Z = [1./el for el in DX]		# Z contains 1/dx, with zeros on diagonal
    Z = Z - np.diag(np.diag(Z))

    D = np.identity(N)
    DM = np.zeros((N,N,M))			# Output matrix, contains 'M' derivatives

    for ell in range(0,M):
        D[:,:] = Z*(C*(np.tile(np.diag(D),(N,1)).T)-D)	
        D[:,:] = [(ell+1)*l for l in D]			# Off-diagonal elements
        trc = np.array([-1.*l for l in np.sum(D.T,0)])
        D = D - np.diag(np.diag(D)) + np.diag(trc)	# Correcting the main diagonal
        DM[:,:,ell] = D

    # Return collocation nodes as a 1-D array
    if not walls:
        return  x[1:-1,0] , np.ascontiguousarray(DM[1:-1, 1:-1])
    else:
        return  x[:,0] , DM



	
def clencurt(Nint,walls=False):
    """ w = clencurt(Nint): Computes the Clenshaw Curtis weights on internal Chebyshev nodes, including a factor of 0.5
    Inputs:
        Nint:  Number of internal Chebyshev collocation nodes
    Outputs:
        w:  1-d numpy array of weights, so that w*f = 0.5 * \int_{-1}^{1} f(y) dy
    To integrate a function, given as a 1-d array of values 'f' on 'N' internal Chebyshev nodes 'y', 
        f_int = np.dot(w,f)"""
    N = Nint + 2
    n1 = N
    if n1 == 1:
        x = 0
        w = 2
    else:
        n = n1 - 1
        C = np.zeros((n1,2),dtype=np.float)
        k = 2*(1+np.arange(int(np.floor(n/2))))
        C[::2,0] = 2/np.hstack((1., 1.-k*k))
        C[1,1] = -n
        V = np.vstack((C,np.flipud(C[1:n,:])))
        F = np.real(ifft(V, n=None, axis=0))
        x = F[0:n1,1]
        w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
    if walls:
        return 0.5*w
    else:
        return 0.5 * w[1:-1]

def _chebdotvec(arr1,arr2,Nint):
    # This function computes inner products of vectors ONLY.
    assert (arr1.ndim == 1) and (arr2.ndim == 1), "For dot products of non-1d-arrays, use chebdot()"

    prod = (np.array(arr1) * np.array(arr2).conjugate() )
    prod = prod.reshape((arr1.size//Nint, Nint))
    wvec = clencurt(Nint).reshape((1,Nint))

    return np.sum( (prod*wvec).flatten() )


def chebnorm(arr,Nint):
    """ norm2 = chebnorm(arr,Nint): Returns 2-norm of array 'arr', weighted by clencurt weights (see clencurt(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of N
                    Can be multi-dimensional.
        Nint:      Number of internal Chebyshev collocation nodes
    Outputs:
        norm:   2-norm, weighted by clenshaw-curtis weights."""
    return np.sqrt(np.abs(_chebdotvec(arr.flatten(),arr.flatten(),Nint) ))


def chebdot(arr1,arr2,Nint):
    """ dotArr = chebdot(arr1, arr2, Nint): Dot products on Chebyshev nodes using clencurt weights
    Inputs:
        arr1: np.ndarray (possibly 2-d) whose size is an integral multiple of Nint
        arr2: np.ndarray (possibly 2-d) whose size is an integral multiple of Nint
            arr1 and arr2 must have same size in their last axis
        Nint: Number of internal Chebyshev nodes
    Outputs:
        dotArr : float of dot product if arr1 and arr2 are 1d
                 2-d array otherwise
        """
    if (arr1.ndim==1) and (arr2.ndim == 1): return _chebdotvec(arr1,arr2,Nint)
    # Nothing special to be done if both are 1d arrays

    # The arguments arr1 and arr2 can either be vectors or matrices
    if (arr1.ndim ==1):
        arr1 = arr1.reshape((1,arr1.size))
    if (arr2.ndim ==1):
        arr2 = arr2.reshape((1,arr2.size))
    dotArr = np.zeros((arr1.shape[0], arr2.shape[0]))
    for ind1 in range(arr1.shape[0]):
        for ind2 in range(arr2.shape[0]):
            dotArr[ind1,ind2] = _chebdotvec(arr1[ind1],arr2[ind1],Nint)
    
    return dotArr 


def chebnorm1(arr,Nint):
    """ norm1 = chebnorm1(arr,Nint): Returns 1-norm of array 'arr', weighted by clencurt weights (see clencurt(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of Nint
                    Can be multi-dimensional.
        Nint:      Number of internal Chebyshev collocation nodes
    Outputs:
        norm:   1-norm, weighted by clenshaw-curtis weights."""
    return _chebdotvec( np.abs(arr.flatten())   , np.ones(arr.size) , Nint) 

def chebnorm2(vec,Nint):
    """ norm2 = chebnorm2(arr,Nint): Returns 2-norm of array 'arr', weighted by clencurt weights (see clencurt(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of Nint
                    If multi-dimensional, arr is flattened.
                    If size > Nint, each 'Nint' elements are normed individually, and their total is added.
        Nint:      Number of internal Chebyshev collocation nodes
    Outputs:
        norm:   2-norm, weighted by clenshaw-curtis weights."""
    return chebnorm(vec,Nint)

def chebnorml(arr,Nint,l=2):
    """ l_norm = chebnorml(arr,Nint): Returns l-norm of array 'arr', weighted by clencurt weights (see clencurt(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of Nint
                    Can be multi-dimensional.
        Nint:      Number of internal Chebyshev collocation nodes
        l(=2):  Order of the norm
    Outputs:
        norm:   l-norm, weighted by clenshaw-curtis weights."""
    return _chebdotvec( np.abs(arr.flatten())**l   , np.ones(arr.size) , Nint) 

def _chebcoeffsvec(f,truncate=True, bVals = [0.,0.]):
    f = f.flatten(); Nint = f.size; N = Nint+2
    fWalled = np.zeros(f.size+2, dtype=f.dtype) # Create an array that also has nodes at the walls
    fWalled[1:-1] = f
    a =  np.fft.fft(np.append(fWalled,fWalled[N-2:0:-1]))  
    a = a[:N]/(N-1.)*np.concatenate(([0.5],np.ones(N-2),[0.5]))
    # Now this is tricky. I actually have info for Nint+2 Chebyshev polynomials, but I don't want to make my code inconsistent.
    # I'll ignore the coefficients for the last two Chebyshev polynomials
    if truncate:
        return a[:-2].astype(f.dtype)
    else:
        return a.astype(f.dtype)


def chebcoeffs(f,truncate=True,**kwargs):
    """coeffs = chebcoeffs(f,truncate=True): Coefficients of Chebyshev polynomials (first kind) for given collocated values.
    Inputs:
        f: Function values on 'Nint' internal Chebyshev nodes (Nint= f.size if f is 1d)
            Can be multi-dimensional
            If f is multidimensional, the last axis size is taken as Nint
        truncate (=True): To ensure size of f and coeffs is consistent, I drop the coefficients for the last two Chebyshev polynomials
                        If truncate is False, retain the last two coefficients and return coeffs with last axis having Nint+2 elements
        bVals (=[0.,0.]): Use this to set non-zero values at the walls for the collocated vector.
    Outputs:
        coeffs: Array of coefficients of Chebyshev polynomials (first kind) from 0 to N-3 (Nint -1)""" 
    if (f.ndim == 1):
        return _chebcoeffsvec(f,truncate=truncate,**kwargs)
    
    shape = list(f.shape)
    Nint= f.shape[-1]
    f = f.reshape((f.size//Nint, Nint))
    N = Nint +2
    if truncate:
        coeffArr = np.zeros((f.shape[0], Nint),dtype=f.dtype)
    else:
        coeffArr = np.zeros((f.shape[0], N), dtype=f.dtype)
        shape[-1] = N
    for ind in range(f.shape[0]):
        coeffArr[ind] = _chebcoeffsvec(f[ind], truncate=truncate)
    coeffArr = coeffArr.reshape(shape)

    return coeffArr


def _chebcoll_vec_vec(a,truncated=True):
    """ Return collocation vector on internal nodes from chebyshev coeffs for a single scalar 
    If truncated is True, it means that the last two coefficients in 'a' have been dropped (see chebcoeffs() )"""
    a = a.flatten()
    Nint = a.size - 2
    if truncated:
        Nint = a.size
        a = np.append(a,[0.,0.]).astype(a.dtype)
    
    N = a.size
    a = a*(N-1.)/np.concatenate(([0.5],np.ones(N-2),[0.5]))

    f = np.fft.ifft(np.append(a,a[N-2:0:-1]))
    
    return f[:Nint].astype(a.dtype)


	
def chebcoll_vec(a,truncated=True):
    """coll = chebcoll_vec(f): Function values on internal Chebyshev nodes, given coefficients of Chebyshev polynomials (first kind).
    Inputs:
        a: Nint Chebyshev polynomial coefficients (Nint= a.size if a is 1d), unless truncated is False
            when truncated is False, 'a' contains Nint+2 polynomial coefficients ('N' coeffs, where 'N' is number of nodes including walls)
            Can be multi-dimensional
            If a is multidimensional, the last axis size is taken as Nint
    Outputs:
        coeffs: Array of function values on Chebyshev nodes, same shape as 'a', except possibly in the last axis, which has Nint elements """ 
    if (a.ndim == 1):
        return _chebcoll_vec_vec(a,truncated=truncated)
   
    shape = list(a.shape)
    Nt= a.shape[-1]
    a = a.reshape((a.size//Nt, Nt))

    if truncated:
        f = np.zeros((a.shape[0],Nt), dtype=a.dtype)
    else:
        f = np.zeros((a.shape[0],Nt-2), dtype=a.dtype)
        shape[-1] = Nt-2

    for ind in range(a.shape[0]):
        f[ind] = _chebcoll_vec_vec(a[ind],truncated=truncated)
    f = f.reshape(shape)
    return f



def chebint(fk, x, walls=False):
    """ interp = chebint(fk,x): Interpolate function 'fk' from internal Chebyshev nodes to 'x'
    Inputs:
        fk: Function values on 'Nint' internal Chebyshev nodes, Nint = fk.size
        x:  Nodes on which f has to be interpolated (in [1,-1])
    Outputs:
        interp: Function values on x
    """
    assert fk.ndim == 1
    if not walls:
        fk = np.concatenate(( [0.], fk, [0.] ))
    speps = np.finfo(float).eps # this is the machine epsilon
    N = np.size(fk)
    M = np.size(x)
    xk = np.sin(np.pi*np.arange(N-1,1-N-1,-2)/(2*(N-1)))
    w = np.ones(N) * (-1)**(np.arange(0,N))
    w[0] = w[0]/2
    w[N-1] = w[N-1]/2
    D = np.transpose(np.tile(x,(N,1))) - np.tile(xk,(M,1))
    D = 1/(D+speps*(D==0))
    p = np.dot(D,(w*fk))/(np.dot(D,w))
    return p



def _chebIntegrateVec(v):
    """ For input 'v' on internal Chebyshev nodes, return \int_{-1}^y v(Y) dY """
    assert v.ndim == 1
    coeffs = chebcoeffs(v, truncate=False)  
    # Get coefficients for Chebyshev polynomials from collocated values.
    # Keep the last two coefficients (truncate=False)

    # Array for integrated coefficients
    Nint = v.size; N = v.size+2
    int_coeffs = np.zeros(N, dtype=coeffs.dtype)

    # T_0 = 1,  T_1 = x, T_2 = 2x^2 -1
    # \int T_0 dx = T_1
    int_coeffs[1] = coeffs[0]

    # \int T_1 dx = 0.25*T_2 + 0.25*T_0
    int_coeffs[2] += 0.25*coeffs[1]
    int_coeffs[0] += 0.25*coeffs[1]

    # \int T_n dx = 0.5*[T_{n+1}/(n+1) - T_{n-1}/(n-1)]
    nvec = np.arange(0,N)
    int_coeffs[3:] += 0.5/nvec[3:]*coeffs[2:N-1]
    int_coeffs[1:N-1] -= 0.5/nvec[1:N-1]*coeffs[2:]

    # Get collocated values from integrated coefficients
    # This is the same as in _chebcoll_vec_vec(), but that function doesn't return the value at y=-1
    # We need the integrated function value at y=-1 to convert into a definite integral
    tmpArr = int_coeffs*(N-1.)/np.concatenate(([0.5],np.ones(N-2),[0.5]))
    tmpArr = np.fft.ifft(np.append(tmpArr,tmpArr[N-2:0:-1]))
    int_coll_vec = tmpArr[:N]

    # Subtract integrated function value at y=-1
    int_coll_vec = int_coll_vec - int_coll_vec[-1]
   
    # Return values only on internal nodes
    return int_coll_vec[1:-1]

def chebIntegrate(v):
    """ int_coll_vec = chebintegrate(v): Integral of function 'v', supposing BC v(y=-1) = 0, i.e. int_coll_vec = \int_{Y=-1}^y  v(Y) dY
    Inputs: 
        v:  Function values on internal Chebyshev grid with Nint=v.size points if v is 1d
            If v is multi-dimensional, Nint = v.shape[-1], and integration is performed on all other axes
    Outputs:
        int_coll_vec: Integral of v, supposing BC v(y=-1) = 0
                        same shape as v""" 
    if v.ndim == 1:
        return _chebIntegrateVec(v)
    
    shape = v.shape
    N = v.shape[-1]

    v = v.reshape((v.size//N, N))
    integral = v.copy()
    for ind in range(v.shape[0]):
        integral[ind] = _chebIntegrateVec(v[ind])
    
    return integral.reshape(shape)




def cheb4c(Nint,returnAll=False):
    """
    Define the fourth order differentiation matrix for channel system incorporating the clamped boundary conditions, 
    u(y=+/-1) = u'(y=+/-1) = 0
    Inputs:
        N:  Number of internal Chebyshev nodes 
        returnAll(bool=False): If true return all four differentiation matrices instead of just the 4th order
    Outputs:
        D4: Fourth order differentiation matrix of size N-2 x N-2 (exclude wall nodes) when returnAll=False
        DM: Differentiation matrix array of size N-2 x N-2 x 4, for D1, D2,D3,D4 when returnAll=True
            
    """
    Nint = np.int(Nint)
    N = Nint+2
    I = np.identity(N-2)

    n1 = np.int(np.floor(N/2. - 1.))
    n2 = np.int(np.ceil(N/2. - 1.))

    k = np.arange(1, N-1).reshape((N-2,1))  
    th = k*np.pi/(N-1.)
    # theta vector (column) 

    x = np.sin( np.pi* np.arange(N-3,2-N,-2)/ (2*(N-1)) )
    x = x.reshape((N-2,1))  # Interior Chebyshev nodes

    s = np.concatenate(( np.sin(th[:n1]), np.sin(th[:n2])[::-1]), axis=0)
   
    # Need to stack row vectors of functions of s in B
    s = s.reshape((1,s.size))
    xrow = x.reshape((1, x.size))

    alpha = s**4
    beta1 = -4.* (s**2) * xrow / alpha
    beta2 =  4.* (3.* (xrow**2) - 1.) / alpha
    beta3 = 24.* xrow / alpha
    beta4 = 24. / alpha
    B = np.vstack( (beta1, beta2, beta3, beta4) )
    
    T = np.tile( th/2., (1, N-2))
    DX = 2. * np.sin(T.T + T) * np.sin(T.T - T)
    DX= np.vstack(( DX[0:n1,:], np.flipud(np.fliplr( [-1.*el for el in DX[0:n2,:]])) ))
    np.fill_diagonal(DX, 1.)  # Fill diagonal with 1s

    # Bring s back to column-vector format
    s = s.reshape((s.size,1))
    ss = (s**2)  * ((-1)**k)
    S  = np.tile(ss, (1, N-2) )
    C  = S / S.T        # C is the matrix with entries c(k)/c(j)

    # Z is matrix with entries 1/(x(k) - x(j)), with zeros on diagonal
    Z = 1. / DX
    np.fill_diagonal(Z, 0.)

    # X is the same as Z.T, but with diagonal entries removed and 
    #   compacted for each column (so number of rows is reduced)
    X = np.zeros((Z.shape[0], Z.shape[1]-1), dtype= Z.dtype)
    X[:] = [np.delete(Z[k], k) for k in range(Z.shape[0])]
    X = X.T

    Y = np.ones((N-3,N-2))
    D = np.identity(N-2)
    DM = np.zeros((N-2,N-2,4), dtype=np.float)

    for ell in range(4):
        ell1 = ell+1
        Y = np.cumsum( np.vstack( (B[ell], ell1*Y[:N-3] * X) ), axis = 0)
        D = ell1 * Z * ( C * np.tile( np.diag(D).reshape((N-2,1)), (1,N-2)) - D)
        D[np.diag_indices_from(D)] = Y[N-3]
        DM[:,:,ell] = D
    if returnAll:
        return DM
    else:
        return DM[:,:,3]


def weightMats(N, walls=False):
    """ Return a dictionary of different weight matrices (see outputs)
    Inputs:
        N: Number of internal collocation nodes
        walls (optional, =False): If True, weight matrices include entries for walls
    Outputs:
        weightDict with keys
            'W2': 2Nx2N weight matrix with clencurt vector on diagonal
            'W3': 3Nx3N, as above
            'W2Sqrt': Similar to W2, but with diagonal elements square-rooted
            'W3Sqrt': 3Nx3N as above
            'W2SqrtInv': Similar to W2Sqrt, but diagonal elements inverted
            'W3SqrtInv': Similar to W3Sqrt, but diagonal elements inverted
        If walls is True, all of the above matrices use Np = N+2 in size instead of N
    """
    w = clencurt(N,walls=walls)
    W1          = np.diag(              w      )
    W2          = np.diag(              np.concatenate((w,w  ))   )
    W3          = np.diag(              np.concatenate((w,w,w))   )
    W1Sqrt      = np.diag(   np.sqrt(   w     ) )
    W2Sqrt      = np.diag(   np.sqrt(   np.concatenate((w,w  )) ) )
    W3Sqrt      = np.diag(   np.sqrt(   np.concatenate((w,w,w)) ) )
    W1SqrtInv   = np.diag(1./np.sqrt(   w     ) )
    W2SqrtInv   = np.diag(1./np.sqrt(   np.concatenate((w,w  )) ) )
    W3SqrtInv   = np.diag(1./np.sqrt(   np.concatenate((w,w,w)) ) )
    weightDict = {'w':w, 'W1':W1, 'W2':W2, 'W3':W3, \
            'W1Sqrt':W1Sqrt, 'W2Sqrt':W2Sqrt, 'W3Sqrt':W3Sqrt,\
            'W1SqrtInv':W1SqrtInv, 'W2SqrtInv':W2SqrtInv, 'W3SqrtInv':W3SqrtInv }
    return weightDict 


#--------------------------------------------------------------------------------
# BL code
# Technically, "Internal nodes" doesn't make sense for a boundary layer.
# Nonetheless, I'll refer to nodes excluding the walls as internal nodes keeping in line with my notation for channel flows

def chebdifBL(Nint,Y=15.):
    """ y, DM = chebdifBL(Nint,Y): Differentiation matrices and mapped Chebyshev nodes for boundary layers
    Inputs:
        Nint:  Number of internal nodes to use in the semi-infinite domain y \in (0,\inf)
        Y:  Scaling factor for the transformation eta = exp(-y/Y)
                A small scaling factor concentrates nodes closer to the wall, which is good for 
                    near-wall resolution, but not good for saturation of derivatives
    Outputs:
        y:      Mapped Chebyshev nodes on (0,\inf)
                    Use exp(-y/Y) to get the corresponding Chebyshev nodes 
        DM:     Differentiation matrices, for four orders.
                Extract as D1 = DM[:,:,0], D2 = [:,:,1] and so on..
    """
    eta,DM = chebdif(2*Nint,4)
    # eta contains internal Chebyshev nodes


    # (0,\inf] in y is mapped to only (1,0) in eta
    eta = eta[:Nint]; DM = np.ascontiguousarray(DM[:Nint,:Nint])

    # Mapped nodes:
    y = -Y*np.log(eta)

    eta = eta.reshape((Nint,1)) # Makes multiplying with 2d arrays easier

    # The differentiation matrices follow from the mapping eta = exp(-y/Y)
    #       Refer to documentation
    D1 = (-1./Y    )*(  eta * DM[:,:,0]  )
    D2 = ( 1./Y**2 )*(  eta * DM[:,:,0] +   (eta**2)*DM[:,:,1]   )
    D3 = (-1./Y**3 )*(  eta * DM[:,:,0] +3.*(eta**2)*DM[:,:,1] +   (eta**3)*DM[:,:,2]  )
    D4 = ( 1./Y**4 )*(  eta * DM[:,:,0] +7.*(eta**2)*DM[:,:,1] +6.*(eta**3)*DM[:,:,2] + (eta**4)*DM[:,:,3]  )

    DMnew = np.zeros((Nint,Nint,4),dtype=np.float)
    DMnew[:,:,0] = D1 
    DMnew[:,:,1] = D2 
    DMnew[:,:,2] = D3 
    DMnew[:,:,3] = D4 

    warn("Differentiation matrices for BLs only work on quantities that go to 0 as y -> inf.")
    
    return y, DMnew

def cheb4cBL(Nint,Y=15.): 
    """ D4 (or DM) = cheb4cBL(Nint,Y=15,returnAll=False):
                Return the fourth differentiation matrix incorporating clamped BCs at y= 0 and y = \inf 
    Inputs:
        Nint:      Number of internal nodes in semi-infinite domain. Number of mapped Cheb nodes (excluding both walls) is 2*Nint
        Y(=15): Scaling factor for transformation. Large factor ensures that function values ignored are low enough. 
    Outputs:
        D4:     Fourth differentiation matrix (returnAll=False) """
    eta= chebdif(2*Nint,4)[0]

    # (0,\inf) is mapped to only (1,0)
    eta = eta[:Nint]    
    
    # Mapped nodes:
    y = -Y*np.log(eta)

    eta = eta.reshape((Nint,1)) # Makes multiplying with 2d arrays easier

    # Define D4 incorporating clamped boundary conditions u = u' = 0 at walls
    DMcl = cheb4c(2*Nint, returnAll=True)  
    DMcl = np.ascontiguousarray(DMcl[:Nint,:Nint])
    D4 = ( 1./Y**4 )*(  eta * DMcl[:,:,0] +7.*(eta**2)*DMcl[:,:,1] +6.*(eta**3)*DMcl[:,:,2] + (eta**4)*DMcl[:,:,3]  )

    return D4





def clencurtBL(Nint,Y=15.):
    """ w = clencurtBL(Nint): Return a weight matrix so that the dot-produce w*f gives the integral of f on (0,inf)
    Inputs:
        Nint:  Number of internal nodes (excluding y=0 and y=inf)
        Y (default: 15):  The scaling factor used in the mapping eta = exp(-y/Y). 
    Outputs:
        w:  Weight matrix
    """
    # The weighting for BLs is supposed work like so:
    # If f(y) is the function whose integral is sought, and ft(eta) its transformed version,
    #   we define a third function gt(eta) = (Y/eta), 
    #   and a regular clencurt weighting of gt should give the integral of f over y. 
    # But, we want to return an array 'w' that can be used for the weighting, 
    #   so, we absorb the factor function (Y/eta) into the new weight matrix defined as wBL = (Y/eta)* wChannel
    eta = chebdif(2*Nint,1)[0]
    wCh = clencurt(2*Nint)
    wCh = wCh[:Nint];   eta = eta[:Nint]    # Only internal nodes > 0

    wBL = (Y/eta)* wCh

    return wBL


def _chebdotvecBL(arr1,arr2,Nint):
    # This function computes inner products of vectors ONLY.
    assert (arr1.ndim == 1) and (arr2.ndim == 1), "For dot products of non-1d-arrays, use chebdotBL()"

    prod = (np.array(arr1) * np.array(arr2).conjugate() )
    prod = prod.reshape((arr1.size//Nint, Nint))
    wvec = clencurtBL(Nint).reshape((1,Nint))

    return np.sum( (prod*wvec).flatten() )


def chebnormBL(arr,Nint):
    """ norm2 = chebnormBL(arr,Nint): Returns 2-norm of array 'arr', weighted by transformed clencurt weights (see clencurtBL(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of Nint
                    Can be multi-dimensional.
        Nint:      Number of nodes in semi-infinite domain, excluding the wall
    Outputs:
        norm:   2-norm, weighted by transformed clenshaw-curtis weights for boundary layers."""
    return np.sqrt(np.abs(_chebdotvecBL(arr.flatten(),arr.flatten(),Nint) ))


def chebdotBL(arr1,arr2,Nint):
    """ dotArr = chebdotBL(arr1, arr2, Nint): Dot products on semi-infinite domain (excluding wall) using transformed clencurt weights
    Inputs:
        arr1: np.ndarray (possibly 2-d) whose size is an integral multiple of Nint
        arr2: np.ndarray (possibly 2-d) whose size is an integral multiple of Nint
            arr1 and arr2 must have same size in their last axis
        Nint: Number of internal nodes on semi-infinite domain (excluding wall)
    Outputs:
        dotArr : float of dot product if arr1 and arr2 are 1d
                 2-d array otherwise
        """
    if (arr1.ndim==1) and (arr2.ndim == 1): return _chebdotvecBL(arr1,arr2,Nint)
    # Nothing special to be done if both are 1d arrays

    # The arguments arr1 and arr2 can either be vectors or matrices
    if (arr1.ndim ==1):
        arr1 = arr1.reshape((1,arr1.size))
    if (arr2.ndim ==1):
        arr2 = arr2.reshape((1,arr2.size))
    dotArr = np.zeros((arr1.shape[0], arr2.shape[0]))
    for ind1 in range(arr1.shape[0]):
        for ind2 in range(arr2.shape[0]):
            dotArr[ind1,ind2] = _chebdotvecBL(arr1[ind1],arr2[ind1],Nint)
    
    return dotArr 



def chebnorm1BL(arr,Nint):
    """ norm1 = chebnorm1BL(arr,Nint): Returns 1-norm of array 'arr', weighted by transformed clencurt weights (see clencurtBL(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of Nint
                    Can be multi-dimensional.
        N:      Number of nodes in semi-infinite domain excluding the wall 
    Outputs:
        norm:   1-norm, weighted by transformed clenshaw-curtis weights for BL."""
    warn("Naive implementation of 1 norm. Rewrite if calling this too many times- use python profiler")
    return _chebdotvecBL( np.abs(arr.flatten())   , np.ones(arr.size) , Nint) 



def chebnorm2BL(vec,Nint):
    """ norm2 = chebnorm2BL(arr,Nint): Returns 2-norm of array 'arr', weighted by transformed clencurt weights for BLs(see clencurtBL(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of Nint
                    If multi-dimensional, arr is flattened.
                    If size > N, each 'N' elements are normed individually, and their total is added.
        Nint:      Number of collocation nodes in semi-infinite domain excluding the wall 
    Outputs:
        norm:   2-norm, weighted by transformed clenshaw-curtis weights for BLs."""
    return chebnormBL(vec,Nint)



def chebnormlBL(vec,Nint,l=2):
    """ l_norm = chebnormlBL(arr,Nint): Returns l-norm of array 'arr', weighted by transformed clencurt weights for BLs (see clencurtlBL(Nint)) 
    Inputs:
        arr:    np.ndarray whose size is an integral multiple of Nint
                    Can be multi-dimensional.
        Nint:      Number of collocation nodes in semi-infinite domain excluding the wall 
        l(=2):  Order of the norm
    Outputs:
        norm:   l-norm, weighted by transformed  clenshaw-curtis weights for BLs."""
    warn("Naive implementation of l-norm. Rewrite if calling this too many times- use python profiler")
    return _chebdotvecBL( np.abs( arr.flatten()**l )   , np.ones(arr.size) , Nint) 




