import unittest
from src.main import *

REPEAT = 1


class Test(unittest.TestCase):

    # facultative, setup the environement before running the tests methods
    def setUp(self):
        pass

    def test_hessenberg(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                H,U = hessenberg(A)
                
                self.assertTrue(is_hessenberg(H))

    def test_hessenberg_transfer_matrix(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                H = A.copy()
                H,U = hessenberg(H)
                
                self.assertTrue(np.max(np.abs(U@H@U.T - A))<PRECISION)

    def test_hessenberg_stable_for_qr_step(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                n = np.shape(A)[0]
                H,U = hessenberg(A)
                hessenberg_qr_step(H,U)

                self.assertTrue(is_hessenberg(H))

    def test_hessenberg_qr_step_transfer_matrix(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                n = np.shape(A)[0]
                H,_ = hessenberg(A)
                U = np.identity(n)
                H2 = H.copy()
                hessenberg_qr_step(H2,U)

                self.assertTrue(np.max(np.abs(U@H2@U.T - H))<PRECISION)

    def test_qr_algo_naive(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                T,U = qr_algo_naive(A)
                self.assertTrue(is_trisup(T))

    def test_qr_algo_naive_transfer_matrix(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                T,U = qr_algo_naive(A)
                self.assertTrue(np.max(np.abs(U@T@U.T - A))<PRECISION)

    def test_qr_algo_hessenberg(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                T,U = qr_algo_hessenberg(A)
                self.assertTrue(is_trisup(T))

    def test_qr_algo_hessenberg_transfer_matrix(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                T = A.copy()
                T,U = qr_algo_hessenberg(T)
                self.assertTrue(np.max(np.abs(U@T@U.T - A)) < PRECISION)

    def test_qr_algo_hessenberg_rayleigh_quotient_shiftl(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                T,U = qr_algo_hessenberg_rayleigh_quotient_shiftl(A)
                self.assertTrue(is_trisup(T))

    def test_qr_algo_hessenberg_rayleigh_quotient_shiftl_transfer_matrix(self):
        for i in range(REPEAT):
            with self.subTest(i=i):
                D = np.diag(np.array([1,2,3,4]))
                S = np.random.uniform(-1,1,(4,4))
                Sinv = np.linalg.inv(S)
                A = np.dot(S,np.dot(D,Sinv))
                T = A.copy()
                T,U = qr_algo_hessenberg_rayleigh_quotient_shiftl(T)
                self.assertTrue(np.max(np.abs(U@T@U.T - A))<PRECISION)

    # facultative, clean up the environement that was setup after running the tests methods
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()

