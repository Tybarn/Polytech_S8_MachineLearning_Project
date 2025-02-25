import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ///////////////////////////////////////////////////////////////////////////////////////////////// #
# -------------------------------------------- FONCTIONS ------------------------------------------ #
# ///////////////////////////////////////////////////////////////////////////////////////////////// #

def calcul_tx_erreur(data, res_obt, res_waited):
    diff_res = np.array(res_obt == res_waited)
    tx_error = (np.array(data[diff_res == False]).shape[0] / data.shape[0] * 100)
    print("Taux d'echec : %.2f\n" % tx_error)
    return tx_error

def dmin(data, res_wished):
    # Trier les images par classe
    w0 = np.array(data[res_wished == 0], dtype=float)
    w1 = np.array(data[res_wished == 1], dtype=float)
    w2 = np.array(data[res_wished == 2], dtype=float)
    w3 = np.array(data[res_wished == 3], dtype=float)
    w4 = np.array(data[res_wished == 4], dtype=float)
    w5 = np.array(data[res_wished == 5], dtype=float)
    w6 = np.array(data[res_wished == 6], dtype=float)
    w7 = np.array(data[res_wished == 7], dtype=float)
    w8 = np.array(data[res_wished == 8], dtype=float)
    w9 = np.array(data[res_wished == 9], dtype=float)

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
    for i in range(data.shape[0]):
        alldist = []
        for j in range(m.shape[0]):
            alldist.append(np.linalg.norm(data[i,:] - m[j,:]))
        classes.append(np.argmin(alldist))

    return classes

# //////////////////////////////////////////////////////////////////////////////////////////// #
# -------------------------------------------- MAIN ------------------------------------------ #
# //////////////////////////////////////////////////////////////////////////////////////////// #

# Chargement des donnees d'entrainement
X_trn = np.load('../trn_img.npy')
Y_trn = np.load('../trn_lbl.npy')

# Chargement des donnees a calculer
X_dev = np.load('../dev_img.npy')
Y_dev = np.load('../dev_lbl.npy')

obj_svc = SVC(gamma='scale')
obj_knn = KNeighborsClassifier(n_neighbors=5)

all_res = np.empty((0,3), dtype=float)
all_times = np.empty((0,3), dtype=float)

i = 1
while i <= 784:
    print(i)
    # Init
    current_res = np.empty((1,3))
    current_times = np.empty((1,3))
    obj_pca = PCA(n_components=i, svd_solver="full")
    obj_pca.fit(X_trn)

    # Entrainement
    obj_svc.fit(obj_pca.transform(X_trn), Y_trn)
    obj_knn.fit(obj_pca.transform(X_trn), Y_trn)

    # Distance minimal
    start_time = time.time()
    dmin_res = dmin(obj_pca.transform(X_dev), Y_dev)
    end_time = time.time()
    current_res[0,0] = (end_time - start_time)
    current_times[0,0] = calcul_tx_erreur(X_dev, dmin_res, Y_dev)

    # SVC
    start_time = time.time()
    svc_res = obj_svc.predict(obj_pca.transform(X_dev))
    end_time = time.time()
    current_res[0,1] = (end_time - start_time)
    current_times[0,1] = calcul_tx_erreur(X_dev, svc_res, Y_dev)

    # Nearest Neighbors
    start_time = time.time()
    knn_res = obj_knn.predict(obj_pca.transform(X_dev))
    end_time = time.time()
    current_res[0,2] = (end_time - start_time)
    current_times[0,2] = calcul_tx_erreur(X_dev, knn_res, Y_dev)

    # Ajout aux matrices contenant tous les resultats
    print(current_res)
    print(current_times)
    all_res = np.append(all_res, current_res, axis=0)
    all_times = np.append(all_times, current_times, axis=0)

    if i == 1:
        i = i + 9
    elif i == 780:
        i = i + 4
    else:
        i = i + 10

# Sauvegarde des resultats dans les fichiers csv
np.savetxt("pca_time.csv", all_times, fmt="%.2f", delimiter=",", newline="\n")
np.savetxt("pca_results.csv", all_res, fmt="%.2f", delimiter=",", newline="\n")