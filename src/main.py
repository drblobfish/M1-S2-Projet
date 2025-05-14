import numpy as np

PRECISION = 1e-10

def givens_cancel_lower_left(k,H):
    norm = np.sqrt(H[k,k]**2 + H[k+1,k]**2)
    return (H[k,k]/norm,
            -H[k+1,k]/norm)

def select_moitie_sous_diagonale(M,n):
    return M[
            np.arange(M.shape[0]) - np.arange(M.shape[1]).reshape(-1,1) < n
            ]

def is_hessenberg(M):
    return (np.abs(select_moitie_sous_diagonale(M, -1)) < PRECISION).all()

def is_trisup(M):
    return (np.abs(select_moitie_sous_diagonale(M, 0)) < PRECISION).all()

def hessenberg_qr_step(H,U):
    n = H.shape[0]
    c = np.zeros(n-1)
    s = np.zeros(n-1)
    for k in range(n-1):
        c[k],s[k] = givens_cancel_lower_left(k,H)
        givens_mat = np.array([[c[k],-s[k]],[s[k],c[k]]])
        H[k:k+2,k:] = givens_mat @ H[k:k+2,k:]
    for k in range(n-1):
        givens_mat = np.array([[c[k],-s[k]],[s[k],c[k]]])
        H[:k+2,k:k+2] = H[:k+2,k:k+2] @ givens_mat.conj().T

        # apply the transform to U
        # careful : U is NOT hessenberg, so you need to multiple the entire column
        U[:,k:k+2] = U[:,k:k+2] @ givens_mat.conj().T
    return

def qr_step_naive(A,U):
    QR_result = np.linalg.qr(A)
    A_new = QR_result.R @ QR_result.Q
    U_new = U @ QR_result.Q
    return A_new,U_new

#Algo permettant de renvoyer la forme de Hessenberg semblable à une matrice A

def hessenberg(A,U):
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

        #update U
        U[0:n,k+1:n] -= 2*((U[0:n,k+1:n]@u))@u.T
    return A,U

def hessenberg_complex(A):
    #Préciser dtype = complex dans la matrice A en entrée
    n = np.shape(A)[0]
    for k in range(n-2):
        x = A[k+1:,k]
        e1 = np.zeros(n-k-1,dtype = complex)
        e1[0] = 1
        a = -np.exp(1j*np.angle(x[0]))*np.linalg.norm(x)
        u = x + a*e1
        u = u.reshape(-1,1)
        u = u/np.linalg.norm(u)
        A[k+1:n,k:n] -= 2*u@(u.conj().T@A[k+1:n,k:n])
        A[0:n,k+1:n] -= 2*((A[0:n,k+1:n]@u))@u.conj().T
    return A

# qr algo with Hessenberg

def qr_algo_naive(A):
    n = A.shape[0]
    U = np.identity(n)
    while not is_trisup(A):
        A,U = qr_step_naive(A,U)
    return A,U

# qr algo with Hessenberg

def qr_algo_hessenberg(A):
    n = A.shape[0]
    U = np.identity(n)
    H,U = hessenberg(A,U)
    while not is_trisup(H):
        hessenberg_qr_step(H,U)
    return H,U

# qr algo with Hessenberg and Rayleigh quotient shift

def qr_algo_hessenberg_rayleigh_quotient_shiftl(A):
    n = A.shape[0]
    U = np.identity(n)
    H,U = hessenberg(A,U)
    for m in range(n-1,0,-1):
        while np.abs(H[m,m-1]) > PRECISION:
            sigma = H[m,m]
            # substract sigma to all term of the diagonal
            H[np.arange(m+1),np.arange(m+1)] -= sigma
            hessenberg_qr_step(H[:m+1,:m+1],U)
            H[np.arange(m+1),np.arange(m+1)] += sigma
    return H,U





def tridiagonale(A):
    # Met sous forme tridiagonale une matrice symétrique réelle A par les transformations de Householder

    n = A.shape[0]

    for k in range(n - 2):
        x = A[k+1:, k]
        e1 = np.zeros(n-k-1,dtype = float)
        e1[0] = 1
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        u = x + alpha * e1
        u = u / np.linalg.norm(u)

        A[k+1:, k+1:] -= 2*((A[k+1:,k+1:]@u))@u.T 
        A[k+1:, k+1:] -= 2*u@(u.T@A[k+1:,k+1:])
        
        A[k+1, k] =  np.linalg.norm(x)
        A[k, k+1] = A[k+1, k]
        
        A[k+2:, k] =  0
        A[k, k+2:] = A[k+2:, k]

    return A

