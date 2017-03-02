import unittest
import numpy as np
from scipy.linalg import norm
from pseudo import *


class pseudoTestCase(unittest.TestCase):
    """ 
    Test functions in pseudospectral methods module "pseudo.py"
    test_chebdifBL(): Using function  f(y) = y^3 exp(-y) + y^6 exp(-5y)
    """
    def test_chebdif(self):
        """ Testing the differentation matrices using a test function,
            f(y) = (1-y**2)**2 + 5(1-y**4)**3
                 = 1 - 2y^2 + y^4 + 5 - 15 y^4 + 15 y^8 - 5 y^12
                This function is chosen to have clamped BCs on [1,-1]
                    so that I can later validate cheb4c with the same funciton
        """
        N = 18
        y, DM = chebdif(N, 4)
        D1 = DM[:,:,0]
        D2 = DM[:,:,1]
        D3 = DM[:,:,2]
        D4 = DM[:,:,3]

        
        f = ( 1. - y**2 )**2 + 5.* ( 1. - y**4 )**3
        # f(y=+/-1) = f'(y=+/-1) = 0
        
        fy = -4.*y - 56.* y**3 + 120.* y**7 - 60.* y**11
        self.assertAlmostEqual( norm( fy - np.dot(D1,f) ) , 0. )

        fy2 = -4. - 168.* y**2 + 840.* y**6 - 660.* y**10
        self.assertAlmostEqual( norm( fy2 - np.dot(D2,f) ) , 0. )

        fy3 = -336.* y + 5040.* y**5 - 6600.* y**9
        self.assertAlmostEqual( norm( fy3 - np.dot(D3,f) ) , 0. )

        fy4 = -336.  + 25200.* y**4 - 59400.* y**8
        self.assertAlmostEqual( norm( fy4 - np.dot(D4,f) ) , 0. )

        return

    def test_cheb4c(self):
        """ Testing the differentation matrices using a test function,
            f(y) = (1-y**2)**2 + 5(1-y**4)**3
                 = 1 - 2y^2 + y^4 + 5 - 15 y^4 + 15 y^8 - 5 y^12
                This function is chosen to have clamped BCs on [1,-1]
                    so that I can later validate cheb4c with the same funciton
        """
        N = 18
        D4 = cheb4c(N)
        y,DM = chebdif(N,4)
        D4_ = DM[:,:,3]
        
        f = ( 1. - y**2 )**2 + 5.* ( 1. - y**4 )**3
        # f(y=+/-1) = f'(y=+/-1) = 0
        

        fy4 = -336.  + 25200.* y**4 - 59400.* y**8
        self.assertAlmostEqual( norm( fy4[1:-1] - np.dot(D4,f[1:-1]) ) , 0. )
        self.assertAlmostEqual( norm( np.dot(D4_, f)[1:-1] - np.dot(D4,f[1:-1]) ) , 0. )

        return

if __name__ == '__main__':
    unittest.main()
