# M1-S2-Projet

Implementation and test of the QR algorithm for Schur Decomposition of matrices.


## Files

```
.
├── doc             documentation
├── Makefile
├── README.md
├── src             code source
├── tests           tests unitaires
└── TODO.md
```

## Usage

```python
>>> from src.main import *
>>> import numpy as np
>>> D = np.diag(np.array([1,2,3,4]))
>>> S = np.random.uniform(-1,1,(4,4))
>>> Sinv = np.linalg.inv(S)
>>> A = S @ D @ Sinv
>>>
>>> qr_algo_hessenberg_rayleigh_quotient_shiftl(A)
>>>
>>> print(np.round(A,3))
[[ 4.     1.34   3.661 -3.28 ]
 [-0.     1.    -5.685 -1.659]
 [ 0.     0.     2.     0.544]
 [ 0.     0.    -0.     3.   ]]
```

## Makefile commands

- pack : creates at time-stamped compressed archive of the whole directory, except useless files
- test : run all unit tests

## Context

This work is produced as a student project for the MA4BY150 class of the MFA master program (University of Paris) under supervision by Aurelie Fisher and Yves Achdou.


