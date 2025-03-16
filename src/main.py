import numpy as np

PRECISION = 1e-10

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
        H[k:k+2,k:] = givens_mat @ H[k:k+2,k:]
    for k in range(n-1):
        givens_mat_t = np.array([[c[k],s[k]],[-s[k],c[k]]])
        H[:k+2,k:k+2] = H[:k+2,k:k+2] @ givens_mat_t

    return H


# test hessenberg_qr_step

def qr_step_naive(A):
    QR_result = np.linalg.qr(A)
    return QR_result.R @ QR_result.Q

#Algo permettant de renvoyer la forme de Hessenberg semblable à une matrice A

def hessenberg(A):
    n = np.shape(A)[0]
    for k in range(n-2):
        x = A[k+1:,k]
        e1 = np.zeros(n-k-1)
        e1[0] = 1
        a = -np.sign(x[0])*np.linalg.norm(x)
        u = x + a*e1
        u = u.reshape(-1,1)
        u = u/np.linalg.norm(u)
        A[k+1:n,k:n] -= 2*u@(u.T@A[k+1:n,k:n])
        A[0:n,k+1:n] -= 2*((A[0:n,k+1:n]@u))@u.T
    return A

# qr algo with Hessenberg and Rayleigh quotient shift

def qr_algo_hessenberg_rayleigh_quotient_shiftl(A):
    n = A.shape[0]
    H = hessenberg(A)
    for m in range(n-1,0,-1):
        while np.abs(H[m,m-1]) > PRECISION:
            sigma = H[m,m]
            # substract sigma to all term of the diagonal
            H[np.arange(m+1),np.arange(m+1)] -= sigma
            hessenberg_qr_step(H[:m+1,:m+1])
            H[np.arange(m+1),np.arange(m+1)] += sigma
    return H

