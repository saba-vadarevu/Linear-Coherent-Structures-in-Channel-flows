""" 
Test for the AMA algorithm to solve the optimization problem by Zare et al. 
The test follows the spring-mass-damper system defined in the original MATLAB code hosted at
http://www.umn.edu/~mihailo/software/ccama/
"""
import numpy as np
import minimize
from scipy.linalg import toeplitz
from scipy.linalg import solve_lyapunov
from numpy.linalg import det
import pdb

# Optimization problem for covariance completion, using a spring-mass-damper system
N = 11  # Number of masses

Ioutput = np.identity(N)    # We'll say the outputs are just the positions of the masses
Istate = np.identity(2*N)   # The state includes positions and velocities of the masses
Zoutput = np.zeros((N,N))

# Defining the dynamic, output, and input matrices
toepArr = np.zeros(N); toepArr[0] = 2.; toepArr[1] = -1.
toepMat = toeplitz(toepArr)
#pdb.set_trace()
# Dynamic matrix
dynMat = np.vstack( ( np.hstack( (Zoutput , Ioutput ) ), 
                      np.hstack( (-toepMat, -Ioutput) ) ) )

# Input matrix
# We don't need this now, but let's get the full thing going..
inMat = np.vstack( (Zoutput, Ioutput) )

# Output matrix
outMat = Istate     # Well what do you know, the output isn't just the position, it's both
# That means Ioutput and Zoutput have inconsistent labels, well.... Let's ignore that.

# Casting all 2D arrays to type matrix 
Ioutput = np.asmatrix(Ioutput); Istate = np.asmatrix(Istate);   Zoutput = np.asmatrix(Zoutput)
dynMat = np.asmatrix(dynMat);   inMat = np.asmatrix(inMat);     outMat = np.asmatrix(outMat)


# Dynamics of the filter that generates colored noise 
#   (I have no idea what's going on here, but hey, if the math works, that's all that matters)
dynMatFil = -Ioutput
inMatFil  = Ioutput
outMatFil = Ioutput
Dfil      = Zoutput  # Don't know what this is...


# Some stuff I haven't figured out yet....
dynMatCas = np.vstack(( np.hstack(( dynMat, inMat*outMatFil)), 
                        np.hstack(( np.zeros((N,2*N)), dynMatFil )) ))
dynMatCas = np.asmatrix(dynMatCas)
inMatCas = np.vstack(( inMat * Dfil, inMatFil ))

# Solving the lyapunov equation for complete covariance matrix of the cascade systems
P = solve_lyapunov( dynMatCas, -inMatCas * inMatCas.H )
P = np.asmatrix(P)

# Covariance of the state of the plant
Sigma = P[:2*N, :2*N]
Sigma = np.asmatrix(Sigma)

#Sigma = -Sigma
#print("Something's messed up here, assigning Sigma to be positive for now..... Revisit this later.")

#--------------------------------------------------------------------------
# The partial covariance subset are diagonal elements on four subblocks

structMat = np.vstack(( np.hstack(( Ioutput, Ioutput )),
                        np.hstack(( Ioutput, Ioutput )) ))
structMat = np.asmatrix(structMat)
# Constraint covariance to be used
covMat = np.multiply( structMat, Sigma )

print("covMat[0,0] is", covMat[0,0])



#---------------------------------------------------------------------------
# Optimization parameters, packing them in a dict

optDict = {}
optDict['rankPar'] = 10.
optDict['stepSize'] = 10.
optDict['tolPrimal'] = 1.0e-06
optDict['tolDual'] = 1.0e-06
optDict['iterMax'] = int(1.0e05)

X0 = solve_lyapunov( dynMat, -Istate)
Z0 = Istate
X0 = np.asmatrix(X0); Z0 = np.asmatrix(Z0)
Y10 = solve_lyapunov( dynMat.H, X0 )
Y10 = optDict['rankPar'] * Y10 / np.linalg.norm(Y10,ord=2)
Y20 = np.identity( outMat.shape[0] )
Y10 = np.asmatrix(Y10); Y20 = np.asmatrix(Y20)

optDict['X0'] = X0
optDict['Z0'] = Z0
optDict['Y10'] = Y10
optDict['Y20'] = Y20

print("Determinants of some matrices... dynMat %.3g, outMat %.3g, covMat %.3g, Y10 %.3g" %( det(dynMat),  det(outMat), det(covMat), det(Y10 ) ))

outDict = minimize.minimize( dynMat, outMat=outMat, structMat=structMat, covMat=covMat, optDict=optDict)

