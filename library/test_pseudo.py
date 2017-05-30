import unittest
import numpy as np
from scipy.linalg import norm
from pseudo import *


class pseudoTestCase(unittest.TestCase):
    """ 
    Test functions in pseudospectral methods module "pseudo.py"
    test_chebdifBL(): Using function  f(y) = y^3 exp(-y) + y^6 exp(-5y)
    """
    print("Norms and dot products have not been tested yet.")
    print();print()
    def test_chebdif(self,N=18):
        """ Testing the differentation matrices using a test function,
            f(y) = (1-y**2)**2 + 5(1-y**4)**3
                 = 1 - 2y^2 + y^4 + 5 - 15 y^4 + 15 y^8 - 5 y^12
                This function is chosen to have clamped BCs on [1,-1]
                    so that I can later validate cheb4c with the same funciton
        """
        print("chebdif... Verifying first 4 derivatives from chebdif using a 12th order polynomial for collocation......")
        N = np.int(N)
        y, DM = chebdif(N, 4)
        #y = y[1:-1]
        #DM = np.ascontiguousarray(DM[1:-1,1:-1])
        D1 = DM[:,:,0]
        D2 = DM[:,:,1]
        D3 = DM[:,:,2]
        D4 = DM[:,:,3]
        
        # Clenshaw-Curtis quadrature for integral norm
        def norm(arr):
            #arr = np.concatenate(([0.],arr,[0.]))
            return chebnorm(arr,N)
        
        
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

    def test_cheb4c(self,N=18):
        """ Testing the differentation matrices using a test function,
            f(y) = (1-y**2)**2 + 5(1-y**4)**3
                 = 1 - 2y^2 + y^4 + 5 - 15 y^4 + 15 y^8 - 5 y^12
                This function is chosen to have clamped BCs on [1,-1]
                    so that I can later validate cheb4c with the same funciton
        """
        print("cheb4c... Comparing clamped D4 from cheb4c with analytical derivative of 12th order polynomial.........")
        N = np.int(N)
        D4 = cheb4c(N)
        y,DM = chebdif(N,4)
        D4_ = DM[:,:,3]
        
        # Clenshaw-Curtis quadrature for integral norm
        def norm(arr):
            #arr = np.concatenate(([0.],arr,[0.]))
            return chebnorm(arr,N)
        
        f = ( 1. - y**2 )**2 + 5.* ( 1. - y**4 )**3
        # f(y=+/-1) = f'(y=+/-1) = 0
        

        fy4 = -336.  + 25200.* y**4 - 59400.* y**8
        self.assertAlmostEqual( norm( fy4 - np.dot(D4,f) ) , 0. )
        self.assertAlmostEqual( norm( np.dot(D4_, f) - np.dot(D4,f) ) , 0. )

        return

    def test_cheb(self,N=18):
        """ Testing if D matrices from chebdif are similar to those from cheb4c(returnAll=True) 
            when the appropriate test functions are used. """
        print("BL... Comparing Dk vs Dk_clamped for channel using an appropriate test function.")
        N = np.int(N)
        y, DM = chebdif(N,4)
        DMcl = cheb4c(N,returnAll=True)
        
        def norm(arr):
            #arr = np.concatenate(([0.],arr,[0.]))
            return chebnorm(arr,N)
        # Clenshaw-Curtis quadrature for integral norm
        
        f = ( 1. - y**2 )**2 + 5.* ( 1. - y**4 )**3
        #f = f[1:-1]
        #DM = DM[1:-1,1:-1]
        normdiff1 = norm( np.dot(DMcl[:,:,0],f)  - np.dot(DM[:,:,0],f) )
        normdiff2 = norm( np.dot(DMcl[:,:,1],f)  - np.dot(DM[:,:,1],f) )
        normdiff3 = norm( np.dot(DMcl[:,:,2],f)  - np.dot(DM[:,:,2],f) )
        normdiff4 = norm( np.dot(DMcl[:,:,3],f)  - np.dot(DM[:,:,3],f) )
        self.assertAlmostEqual( normdiff1 , 0. )
        self.assertAlmostEqual( normdiff2 , 0. )
        self.assertAlmostEqual( normdiff3 , 0. )
        self.assertAlmostEqual( normdiff4 , 0. )
        print("Norm diff comparing D1 to D1cl:", normdiff1)
        print("Norm diff comparing D2 to D2cl:", normdiff2)
        print("Norm diff comparing D3 to D3cl:", normdiff3)
        print("Norm diff comparing D4 to D4cl:", normdiff4)
       
        return



    def test_chebdifBL(self, N=121, Y=15.,tol=1.0e-05):
        """ Testing the differentation matrices for BL using a test function,
            f(y) =  y**2 * e^{-2y}
                This function is chosen to have clamped BCs at y = 0, \inf
                    so that I can later validate cheb4cBL with the same funciton
        """
        print("chebdifBL.... Dk vs function-derivatives for BL, k in [1,4]")
        N = np.int(N)
        y, DM = chebdifBL(N,Y=Y)
        D1 = DM[:,:,0]; D2 = DM[:,:,1]; D3 = DM[:,:,2]; D4 = DM[:,:,3]

        norm = lambda arr: chebnormBL(arr,N)
        # Integral norm on the semi-infinite domain

        # Test function and its derivatives
        f   = np.exp(-2.*y) * (y**2) 

        fy  = np.exp(-2.*y) * (2.*y - 2.*y**2)
        self.assertLess( np.abs(norm( fy - np.dot(D1,f) )) , tol)
        print("Norm diff for D1 is ", np.abs(norm( fy - np.dot(D1,f) )))

        fy2 = np.exp(-2.*y) * (2. - 8.*y + 4.*y**2 )
        self.assertLess( np.abs(norm( fy2 - np.dot(D2,f) )) , tol)
        print("Norm diff for D2 is ", np.abs(norm( fy2 - np.dot(D2,f) )))

        fy3 = np.exp(-2.*y) * (-12. + 24.*y -8.*y**2)
        self.assertLess( np.abs(norm( fy3 - np.dot(D3,f) )) , tol)
        print("Norm diff for D3 is ", np.abs(norm( fy3 - np.dot(D3,f) )))

        fy4 = np.exp(-2.*y) * (48. - 64.*y + 16.*y**2)
        print("Norm diff for D4 is ", np.abs(norm( fy4 - np.dot(D4,f) )))
        self.assertLess( np.abs(norm( fy4 - np.dot(D4,f) )) , 50.*tol)
        # The derivatives get less accurate as the order increases
        # Here we're down to just 10^-3 for fourth order, which is a shame really

        print("Norm diff comparing D2 to D1*D1:", norm(np.dot(D2,f)- np.dot(D1,np.dot(D1,f))))
        print("Norm diff comparing D3 to D2*D1:", norm(np.dot(D3,f)- np.dot(D2,np.dot(D1,f))))
        print("Norm diff comparing D4 to D3*D1:", norm(np.dot(D4,f)- np.dot(D3,np.dot(D1,f))))
        print("Norm diff comparing D3 to D1*D2:", norm(np.dot(D3,f)- np.dot(D1,np.dot(D2,f))))
        print("Norm diff comparing D4 to D1*D3:", norm(np.dot(D4,f)- np.dot(D1,np.dot(D3,f))))


        return


    def test_chebBL(self, N=121,Y=15., tol=1.0e-05):
        """ Comparing differentiation matrices from chebdif to those from cheb4c
        for a test function that satisfies clamped BCs in the BL:
            f(y) = y^2 * exp(-2y)"""
        print("chebBL... Dk vs Dk_cl vs fyk for BL")
        N = np.int(N)
        eta = chebdif(2*N+2,1)[0]
        DMchcl = cheb4c(2*N+2,returnAll=True)
        # [0,\inf] is mapped to only [1,0)
        eta = eta[1:N+1] 
        DMchcl = np.ascontiguousarray(DMchcl[:N,:N])
        # Ignoring the wall at y = 0 or eta = 1
       
        norm = lambda arr: chebnormBL(arr,N)
        # Integral norm on the semi-infinite domain

        # Mapped nodes:
        y = -Y*np.log(eta)

        eta = eta.reshape((N,1)) # Makes multiplying with 2d arrays easier

        # Test function:
        f = (y**2)* np.exp(-2.*y)
        fy1 = np.exp(-2.*y) * (2.*y - 2.*y**2)
        fy2 = np.exp(-2.*y) * (2. - 8.*y + 4.*y**2)
        fy3 = np.exp(-2.*y) * (-12. + 24.*y - 8.*y**2)
        fy4 = np.exp(-2.*y) * (48.  - 64.*y + 16.*y**2)

        # The differentiation matrices follow from the mapping eta = exp(-y/Y)
        #       Refer to documentation
        D1 = (-1./Y    )*(  eta * DMchcl[:,:,0]  )
        D2 = ( 1./Y**2 )*(  eta * DMchcl[:,:,0] +   (eta**2)*DMchcl[:,:,1]   )
        D3 = (-1./Y**3 )*(  eta * DMchcl[:,:,0] +3.*(eta**2)*DMchcl[:,:,1] +   (eta**3)*DMchcl[:,:,2]  )
        D4 = ( 1./Y**4 )*(  eta * DMchcl[:,:,0] +7.*(eta**2)*DMchcl[:,:,1] +6.*(eta**3)*DMchcl[:,:,2] + (eta**4)*DMchcl[:,:,3]  )


        y_, DM_ = chebdifBL(N,Y=Y)
        print("Diff between y from chebdifBL and from calculation in test function is ", norm(y - y_))


        print("Norm diff between D1 and D1cl is ", np.abs(norm( np.dot(D1,f) - np.dot(DM_[:,:,0],f) )))
        print("Norm diff between fy1 and D1cl is ", np.abs(norm( np.dot(D1,f) - fy1 )))
        self.assertLess( np.abs(norm( np.dot(D1,f) - np.dot(DM_[:,:,0],f)  )) , tol)
        self.assertLess( np.abs(norm( np.dot(D1,f) - fy1  )) , tol)

        print("Norm diff between D2 and D2cl is ", np.abs(norm( np.dot(D2,f) - np.dot(DM_[:,:,1],f) )))
        print("Norm diff between fy2 and D2cl is ", np.abs(norm( np.dot(D2,f) - fy2 )))
        self.assertLess( np.abs(norm( np.dot(D2,f) - np.dot(DM_[:,:,1],f)  )) , tol)
        self.assertLess( np.abs(norm( np.dot(D2,f) - fy2  )) , tol)
        
        print("Norm diff between D3 and D3cl is ", np.abs(norm( np.dot(D3,f) - np.dot(DM_[:,:,2],f) )))
        print("Norm diff between fy3 and D3cl is ", np.abs(norm( np.dot(D3,f) - fy3 )))
        self.assertLess( np.abs(norm( np.dot(D3,f) - np.dot(DM_[:,:,2],f)  )) , tol)
        self.assertLess( np.abs(norm( np.dot(D3,f) - fy3  )) , tol)
       
        print("Norm diff between D4 and D4cl is ", np.abs(norm( np.dot(D4,f) - np.dot(DM_[:,:,3],f) )))
        print("Norm diff between fy4 and D4cl is ", np.abs(norm( np.dot(D4,f) - fy4 )))
        self.assertLess( np.abs(norm( np.dot(D4,f) - np.dot(DM_[:,:,3],f)  )) , tol)
        self.assertLess( np.abs(norm( np.dot(D4,f) - fy4  )) , tol)

        return

    def test_clencurtBL(self,N=121,Y=15.):
        """ Ensure that the weight array gives the integral norm"""
        pass

if __name__ == '__main__':
    unittest.main()
