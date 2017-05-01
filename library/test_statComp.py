import numpy as np
import sys
import pseudo
import os
import miscUtil
import ops


specDataDir = os.environ['DATA'] + 'spec/'
covDataDir = os.environ['DATA'] + 'cov/'
libDir = os.environ['LINLIB']


##=======================================================
# Load mean turbulent velocity from DNS, and interpolate 
N = 64

U0 = np.load(specDataDir+'uMeanN384.npy')
U = np.zeros(U0.size+2)  # Add wall-nodes
U[1:-1] = U0


# Interpolate to a grid with N=64 (internal nodes)
Uc = pseudo.chebcoeffs(U)
Uc = Uc[:N+2]
U = pseudo.chebcoll_vec(Uc)
U = U[1:-1] # Lose the wall-nodes

##========================================================
# Initialize statComp instance

a = 2.; b = 8.; l = a; m = b/2.
statInst = ops.statComp(a=a, b=b, N=N)
# Re-assign attributes U, dU, d2U to reflect turbulent field
statInst.U = U
statInst.dU = statInst.D1 @ U
statInst.d2U = statInst.D2 @ U


##==========================================================
# Iterate for stat completion
outStats = statInst.completeStats(
        iterMax=10000,
        tolPrimal=1.0e-03,
        tolDual=1.0e-03,
        savePath='../')
