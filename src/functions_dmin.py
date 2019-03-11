import numpy as np

def dmin(X, Y):
    # Trier les images par classe
    w0 = np.array(X[Y == 0], dtype=float)
    w1 = np.array(X[Y == 1], dtype=float)
    w2 = np.array(X[Y == 2], dtype=float)
    w3 = np.array(X[Y == 3], dtype=float)
    w4 = np.array(X[Y == 4], dtype=float)
    w5 = np.array(X[Y == 5], dtype=float)
    w6 = np.array(X[Y == 6], dtype=float)
    w7 = np.array(X[Y == 7], dtype=float)
    w8 = np.array(X[Y == 8], dtype=float)
    w9 = np.array(X[Y == 9], dtype=float)

    # Baricentres de chaque classe
    m = np.average(w0, axis=0)
    m = np.vstack((m, np.average(w1, axis=0)))
    m = np.vstack((m, np.average(w2, axis=0)))
    m = np.vstack((m, np.average(w3, axis=0)))
    m = np.vstack((m, np.average(w4, axis=0)))
    m = np.vstack((m, np.average(w5, axis=0)))
    m = np.vstack((m, np.average(w6, axis=0)))
    m = np.vstack((m, np.average(w7, axis=0)))
    m = np.vstack((m, np.average(w8, axis=0)))
    m = np.vstack((m, np.average(w9, axis=0)))

    # Calcul des distances minimales de chaque image par rapport aux baricentres
    classes = []
    for i in range(X.shape[0]):
        alldist = []
        for j in range(m.shape[0]):
            alldist.append(np.linalg.norm(X[i,:] - m[j,:]))
        classes.append(np.argmin(alldist))
    #res = np.array(classes == Y)
    #print("Taux d'erreur : ")
    #print(np.array(X[res == False]).shape[0] / X.shape[0] * 100)

    return classes
