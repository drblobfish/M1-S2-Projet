import unittest
from src.main import *


class Test(unittest.TestCase):

    # facultative, setup the environement before running the tests methods
    def setUp(self):
        pass

    def is_hessenberg(self,M):
        return (np.abs(
            M[np.arange(M.shape[0]) -
              np.arange(M.shape[1]).reshape(-1,1)
              < -1]
            )
                < PRECISION).all()

    def is_trisup(self,M):
        return (np.abs(
            M[np.arange(M.shape[0]) -
              np.arange(M.shape[1]).reshape(-1,1)
              < 0]
            )
                < PRECISION).all()


    def test_hessenberg(self):
        for i in range(50):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                H = hessenberg(A)
                
                self.assertTrue(self.is_hessenberg(H))

    def test_hessenberg_stable_for_qr_step(self):
        for i in range(50):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                H = hessenberg(A)
                H2 = hessenberg_qr_step(H)
                self.assertTrue(self.is_hessenberg(H2))

    def test_qr_algo(self):
        for i in range(50):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                qr_algo_hessenberg_rayleigh_quotient_shiftl(A)
                self.assertTrue(self.is_trisup(A))

    # facultative, clean up the environement that was setup after running the tests methods
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()

