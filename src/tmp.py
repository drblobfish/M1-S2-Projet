D = np.diag(np.array([1,2,3,4]))
S = np.random.uniform(-1,1,(4,4))
Sinv = np.linalg.inv(S)
A = np.dot(S,np.dot(D,Sinv))

H = A.copy()
U_1 = np.identity(A.shape[0])
H,U_1 = hessenberg(H,U_1)


T = H.copy()
U_2 = np.identity(A.shape[0])
hessenberg_qr_step(T,U_2)

T_2 = H.copy()
U_3 = U_1.copy()
hessenberg_qr_step(T_2,U_3)

