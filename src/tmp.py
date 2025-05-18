from src.main import *

n = 4
D = np.diag(np.array([1,2,3,4]))
S = np.random.uniform(-1,1,(4,4))
Sinv = np.linalg.inv(S)
A = np.dot(S,np.dot(D,Sinv))

H = A.copy()
U_1 = np.identity(A.shape[0])
H,U_1 = hessenberg(H)

np.round(A,3)
np.round(U_1@H@U_1.T,3)

np.round(H,3)
sigma = H[-1,-1]
H_shift = H.copy()
H_shift[np.arange(n),np.arange(n)] -= sigma
np.round(H_shift,3)

T_shift = H_shift.copy()
U_2 = np.identity(n)
hessenberg_qr_step(T_shift,U_2)

np.round(U_2@T_shift@U_2.T,3)
T = T_shift.copy()
T[np.arange(n),np.arange(n)] += sigma
np.round(U_2@T@U_2.T,3)
np.round(H,3)


