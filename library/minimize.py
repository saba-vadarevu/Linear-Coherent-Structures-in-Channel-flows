"""
Python translation for the MATLAB code implementing AMA and ADMM by Armin Zare and Mihailo Jovanovic (http://www.umn.edu/~mihailo/software/ccama/)

This code soloves the minimization problem
minimize  	-log(|X|) + gamma ||Z||
subject to 	A X + X A' + Z = 0
                E .* X - G = 0       (E .* X is elementwise product)


for the state evolution equation
    dx/dt  = Ax + Bu,   A is the dynamic matrix, B is the forcing matrix
with the output
    y = Cu,             C is the output matrix
Here, A' refers to the conjugate of A

X is the state-covariance, X = expect( x x'),
G is a matrix of covariances available from experiment/computation
E is the structural identity such that 
        E_ij =  1       if G_ij is available
                0       if G_ij is not available
Z is the matrix appearing in the Lyapunov equation for state covariance,
    AX + XA' + Z = 0
gamma is a rank parameter weighting the importance of minimizing the rank of Z
"""
import numpy as np
from scipy.linalg import solve_lyapunov
from warnings import warn


def minimize(dynMat, outMat = None, structMat = None, covMat =None, rankPar=10.,optDict=None):
    """
        Minimizes   -log(|X|) + rankPar  || Z|| 
        subject to  dynMat * X + X * dynMat' + Z = 0
                    structMat .* x - covMat = 0

        Outputs:
            outDict with keys:
            X, Z, Y1, Y2
    """
    #-----------------------------------------------------------------
    # Initialization and Pre-processing..............................
    assert dynMat.ndim == 2
    assert covMat is not None, "I need some statistics to run this optimization problem...."
    assert covMat.ndim == 2

    norm = np.linalg.norm   # Makes life easier later

    # Converting all input arrays into matrices
    # IMPORTANT: * NOW REPRESENTS MATRIX MULTIPLICATION AND NOT ELEMENTWISE MULTIPLICATION
    warn("Be careful with array multiplication. All arrays are now matrices, so A*B is matrix multiplication, not elementwise multiplication.")
    
    # Dimensionality of the state 
    nState = dynMat.shape[0]
    

    if outMat is None:
        outMat = np.identity(nState, dtype=dynMat.dtype)
        # Take the state as the output if not specified
    # Dimensionality of output
    nOut = outMat.shape[0]


    
    if structMat is None:
        # Treat all zero entries of covMat as being unknown.
        # It's not ideal, but hey, if you're too lazy to supply a structMat....
        structMat = covMat.copy().astype(bool).astype(np.int)
    
    dynMat = np.asmatrix(dynMat); outMat = np.asmatrix(outMat)
    covMat = np.asmatrix(covMat); structMat = np.asmatrix(structMat)

    # Default values for iterative solution of the optimization problem
    # Documentation on the iterations to be added soon
    stepSize = optDict.get("stepSize",10.)
    tolPrimal = optDict.get("tolPrimal",1.0e-06)
    tolDual = optDict.get("tolDual",1.0e-06)
    iterMax = optDict.get("iterMax",1.0e05)

    # Initializing X, Z, Y1, and Y2 for the iterations
    # Documentation coming soon

    Z0 = optDict.get("Z0",None); X0 = optDict.get("Z0",None); Y10 = optDict.get("Y10",None); 
    if Z0 is None:
        Z0 = np.identity(nState , dtype=dynMat.dtype )
    if X0 is None:
        X0 = solve_lyapunov( dynMat, Z0 )
    X0 = np.asmatrix(X0);   Z0 = np.axmatrix(Z0)

    if Y10 is None:
        Y10 = solve_lyapunov( dynMat.getH(), -X0)
        if not hasattr(Y10, "H"):
            warn("Y10 from solve_lyapunov does not have attribute 'H', so it's an ndarray and not a matrix. I've recast it into matrix, but have a look at why this is happening....")
        Y10 = np.asmatrix(Y10)
        Y10 = rankPar * Y10 / norm(Y10,2) # Norm returns largest singular value
    else:
        Y10 = np.asmatrix(Y10)

    # Y2 relates to imposing the constraint due to observed covariance stats,
    #   so it's size should relate to the dimensionality of the output (not state)
    if Y20 is None:
        Y20 = np.identity( outMat.shape[0], dtype=dynMat.dtype )
    Y20 = np.asmatrix(Y20)
    X = X0; Z = Z0; Y1 = Y10; Y2 = Y20
    
    stepSize0 = stepSize
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    # AMA method for optimization:
    print("Write decorators for timing and logging (have a look at your bookmarks on Chrome)")
    warn("This code currently uses only method AMA. ADMM from the original matlab function hasn't been implemented.")

    # Ignoring some lines from original MATLAB code because I don't see why they're needed (yet)

    Istate = np.identity(nState, dtype=dynMat.dtype)    # Identity
    beta = 0.5      # Backtracking parameter
    stepSize1 = stepSize0   # Initial step size
    failIters = []  # Log failed iteration numbers

    print(""); print("Starting iterations for AMA......")
    print("stepSize_BB  stepSize  tolPrimal   resPrimal  tolDual   abs(eta)    iter")

    funPrimalArr = np.zeros(iterMax)
    funDualArr = np.zeros(iterMax)
    resPrimalArr = np.zeros(iterMax)
    dualGapArr = np.zeros(iterMax)
    # Will include time later



    for AMAstep in range(iterMax):

        # Minimization step for X^{k+1}
        Xnew = np.linalg.solve( dynMat.H * Y1 + Y1 * dynMat 
                +  outMat.H * np.multiply( structMat * Y2) * outMat,  Istate )
        # To get the inverse of a matrix A, apparently np.linalg.solve(A, Identity) is better than np.linalg.inv(A)
        # I'm not sure yet if I need to call this Xnew instead of just X, I'll get back to this later

        Xnew = (Xnew + Xnew.H)/2.

        eigX = np.real( np.linalg.eig(Xnew) )
        logDetX = np.sum( np.log( eigX ) ) 



        # Gradient of the dual function
        gradD1 = dynMat * Xnew  +  Xnew * dynMat.H
        gradD2 = np.multiply( structMat, outMat * Xnew * outMat.H) - covMat

        
        # Define the primal residual as difference in new statistics from observed (constraints)
        resPrimalNew2 = gradD2
        stepSize = stepSize1



        for innerIter in range(50):
            a = rankPar/stepSize

            # Minimization step for Z^{k+1} 
            Wnew = -( dynMat * Xnew + Xnew * dynMat.H  +  (1./stepSize) * Y1 )
            U, svals , V = np.linalg.svd(Wnew)
            # The above line differs from the original matlab code because numpy's svd returns the singular values as an array directly, while Matlab returns a diagonal matrix
            if isinstance(svals,np.matrix):
                warn("The array of singular values is a matrix and not a regular array")
                svals = svals.A
            assert isinstance(U, np.matrix) and isinstance(V, np.matrix)

            # Singular value thresholding
            svalsNew = (  ( 1. - a/np.abs(svals) ) * svals)  * (np.abs(svals) > a).astype(int)
            # So what happened there is, for svals>a, sNew = (1 - a/|s|) * s = s-aA
            #   That doesn't make much sense to me, but we'll see how it goes
            #   Singular values should always be positive, so I don't see why all of that extra stuff is needed...

            # Update Z
            Znew = U * np.asmatrix(np.diag(svalsNew)) * V.H
            Znew = (Znew + Znew.H)/2.



            # Update Y
            resPrimalNew1 = gradD1 + Znew


            # Lagrange multiplier update
            Y1new = Y1 + stepSize * resPrimalNew1
            Y2new = Y2 + stepSize * resPrimalNew2
            Y1new = (Y1new + Y1new.H)/2.    # Keep them Hermitian
            Y2new = (Y2new + Y2new.H)/2.

            assert isinstance(Y1new, np.matrix) and (Y2new, np.matrix)

            # Eigenvalues of X^{-1} (why though?) 
            evalsLadjYnew = np.real( np.linalg.eigvals( dynMat.H * Y1new + Y1new * dynMat 
                + outMat.H * np.multiply( structMat, Y2new ) * outMat  ) )
            logDetLadjYnew = np.sum( np.log( evalsLadjYnew ))
            dualYnew = logDetLadjYnew - np.trace( G * Y2new ) + nState

            if np.amin( evalsLadjYnew ) < 0.:
                stepSize *= beta    # Backtrack the step
            elif ( dualYnew < dualY + \
                            np.trace( gradD1 *(Y1new - Y1)) + \
                            np.trace( gradD2 *(Y2new - Y2)) - \
                            (0.5/stepSize) * norm(Y1new - Y1, ord='fro')**2 - \
                            (0.5/stepSize) * norm(Y2new - Y2, ord='fro')**2 ):
                # Note: Frobenius norm essentially flattens the array and calculates a vector 2-norm
                stepSize *= beta
            else:
                break
                # So this is where we're breaking out of the inner loop
                # I suppose what's happening here is we're checking if the primal residual is acceptable
            

        # Primal residual
        resPrimalNew1 = dynMat * Xnew + Xnew * dynMat.H + Znew
        resPrimalNew2 = np.multiply( structMat, outMat * Xnew * outMat.H ) - covMat
        resPrimal = np.sqrt( norm( resPrimalNew1, ord='fro')**2 + norm( resPrimalNew2, ord='fro')**2  )


        # Calculating the duality gap
        dualGap = - logDetX + rankPar * np.sum( svalsNew) - dualYnew

        # Print progress for every 100 outer iterations
        if AMAstep%100 == 0:
            print("%12.2g  %10.2g  %10.2g  %10.2g  %10.2g  %10.2g  %d" %(stepSize1, \
                    stepSize, tolPrimal, resPrimal, tolDual, np.abs(dualGap), AMAstep) )


        # We start BB stepsize selection now, apparently, whatever that is.. 
        Xnew1 = np.linalg.solve( \
                dynMat.H * Y1new + Y1new * dynMat + outMat.H * np.multiply( structMat, Y2new) * outMat, \
                Istate)
        Xnew1 = ( Xnew1 + Xnew1.H )/2.
        gradD1new = dynMat * Xnew1 + Xnew1 * dynMat.H
        gradD2new = np.multiply( structMat, outMat * Xnew1 * outMat.H) - G
        stepSize1 = np.real(  ( norm(Y1new - Y1, ord='fro')**2 + norm(Y2new - Y2, ord='fro')**2 )/ \
                ( np.trace( (Y1new - Y1) * (gradD1 - gradD1new) ) + np.trace( (Y2new - Y2) * (gradD2 - gradD2new) )   ) )

        if (stepSize1 < 0.) or not np.isfinite(stepSize1):
            stepSize1 = stepSize0
            failIters.append(AMAstep)

        # This is the end of the iteration... Apparently
        funPrimalArr[AMAstep] = -np.log( np.linalg.det( Xnew )) + rankPar * norm(Znew,ord='nuc')
        funDualArr[AMAstep] = np.real(dualYnew)
        resPrimalArr[AMAstep] = resPrimal
        dualGapArr[AMAstep] = np.abs(dualGap)
        # Again, I'll have to include execution time info later


        # Stopping criteria
        if ( np.abs(dualGap) < tolDual) and (resPrimal < tolPrimal):
            print("AMA converged to assigned accuracy!")
            print("AMA steps: %d" %AMAstep)
            print("%12.2g  %10.2g  %10.2g  %10.2g  %10.2g  %10.2g  %d" %(stepSize1, \
                    stepSize, tolPrimal, resPrimal, tolDual, np.abs(dualGap), AMAstep) )
            break

        Y1 = Y1new
        Y2 = Y2new
        dualY = dualYnew

    # Assigning an output dictionary
    outDict = {}
    if AMAstep == iterMax:
        outDict['flag'] = 0
    else:
        outDict['flag'] = 1

    outDict['X'] = Xnew
    outDict['Z'] = Znew
    outDict['Y1'] = Y1new
    outDict['Y2'] = Y2new
    outDict['steps'] = AMAstep
    outDict['funPrimalArr'] = funPrimalArr[:AMAstep]
    outDict['funDualArr'] = funDualArr[:AMAstep]
    outDict['dualGapArr'] = dualGapArr[:AMAstep]

    return outDict















            
