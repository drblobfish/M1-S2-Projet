import numpy as np
from scipy.linalg import hessenberg


# generate A
D = np.diag(np.array([1,2,3,4]))
S = np.random.uniform(-1,1,(4,4))
Sinv = np.linalg.inv(S)
A = np.dot(S,np.dot(D,Sinv))


# put A in hessenberg form
H,Q = hessenberg(A,calc_q=True)

def givens_cancel_lower_left(k,H):
    norm = np.sqrt(H[k,k]**2 + H[k+1,k]**2)
    return (H[k,k]/norm,
            -H[k+1,k]/norm)


def hessenberg_qr_step(H):
    n = H.shape[0]
    c = np.zeros(n-1)
    s = np.zeros(n-1)
    for k in range(n-1):
        c[k],s[k] = givens_cancel_lower_left(k,H)
        givens_mat = np.array([[c[k],-s[k]],[s[k],c[k]]])
        H[k:k+2,k:] = np.dot(givens_mat,H[k:k+2,k:])
    for k in range(n-1):
        givens_mat_t = np.array([[c[k],s[k]],[-s[k],c[k]]])
        H[:k+2,k:k+2] = np.dot(H[:k+2,k:k+2],givens_mat_t)

    return H


# test hessenberg_qr_step

def qr_step_naive(A):
    QR_result = np.linalg.qr(A)
    return np.dot(QR_result.R,QR_result.Q)

test_hessenberg_1 = hessenberg(np.random.uniform(-1,1,(5,5)))
test_hessenberg_2 = test_hessenberg_1.copy()

hessenberg_qr_step(test_hessenberg_1)
qr_step_naive(test_hessenberg_2)

# qr algo with Hessenberg and Rayleigh quotient shift

def qr_algo_hessenberg_rayleigh_quotient_shiftl(A):
    n = A.shape[0]
    H = hessenberg(A)
    print(np.round(H,3))
    for m in range(n-1,0,-1):
        while np.abs(H[m,m-1]) > 1e-5:
            sigma = H[m,m]
            # substract sigma to all term of the diagonal
            H[np.arange(m+1),np.arange(m+1)] -= sigma
            hessenberg_qr_step(H[:m+1,:m+1])
            H[np.arange(m+1),np.arange(m+1)] += sigma
        print(np.round(H,3))
    return H
