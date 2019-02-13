import numpy as np
import matplotlib.pyplot as plt


def dmin :
    w1 = np.array([[0,0,1,1],[0,1,0,1]], dtype=float)
    w2 = np.array([[2,2,3,3],[0,1,0,1]], dtype=float)

    n1 = 4
    n2 = 4

    m1 = np.average(w1,axis=1)
    m2 = np.average(w2,axis=1)

    A = m2 - m1
    m = (m1+m2)/2

    R1 = np.dot(A, (w1 - m))
    R2 = np.dot(A, (w2 - m))