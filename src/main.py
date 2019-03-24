import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ///////////////////////////////////////////////////////////////////////////////////////////////// #
# -------------------------------------------- FONCTIONS ------------------------------------------ #
# ///////////////////////////////////////////////////////////////////////////////////////////////// #

def calcul_tx_erreur(data, res_obt, res_waited):
    diff_res = np.array(res_obt == res_waited)
    tx_erreur = (np.array(data[diff_res == False]).shape[0] / data.shape[0] * 100)
    print("Taux d'erreur : %.2f\n" % tx_erreur)
    return tx_erreur

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
X_trn = np.load('../data/trn_img.npy')
Y_trn = np.load('../data/trn_lbl.npy')

# Chargement des donnees a calculer
X_dev = np.load('../data/dev_img.npy')
Y_dev = np.load('../data/dev_lbl.npy')

# Chargement des donnees a estimer pour la note
X_tst = np.load('../data/tst_img.npy')


# ------------------------ Question 1 : DMIN ------------------------

print(" -------------------- DMIN ---------------------\n")

# Estimation
start_time = time.time()
dmin_res = dmin(X_dev, Y_dev)
end_time = time.time()

print("Temps d'execution : %.2f sec" % (end_time - start_time))
calcul_tx_erreur(X_dev, dmin_res, Y_dev)


# ------------------------ Question 2 : PCA + DMIN ------------------------

print(" -------------------- PCA + DMIN ---------------------\n")

# Creation et entrainement de la pca
obj_pca = PCA(n_components=784)
obj_pca.fit(X_trn)

# Estimation
start_time = time.time()
pca_dmin_res = dmin(obj_pca.transform(X_dev), Y_dev) # Calcul des donnees modifiees par la pca pour dmin
end_time = time.time()

print("Temps d'execution : %.2f sec" % (end_time - start_time))
calcul_tx_erreur(X_dev, pca_dmin_res, Y_dev)


# ------------------------ Question 3 : SVM ------------------------

print(" -------------------- SVC ---------------------\n")

# Creation de l'objet SVC et entrainement
obj_svc = SVC(gamma='scale')
obj_svc.fit(X_trn, Y_trn)

# Estimation
start_time = time.time()
svc_res = obj_svc.predict(X_dev)
end_time = time.time()

print("Temps d'execution : %.2f sec" % (end_time - start_time))
calcul_tx_erreur(X_dev, svc_res, Y_dev)


# ----------------- Question 3 : Nearest Neighbors -----------------

print(" -------------------- Nearest Neighbors --------------------\n")

# Creation du classifieur neighbors et entrainement
obj_knn = KNeighborsClassifier(n_neighbors=5)
obj_knn.fit(X_trn, Y_trn)

# Estimation
start_time = time.time()
knn_res = obj_knn.predict(X_dev)
end_time = time.time()

print("Temps d'execution : %.2f sec" % (end_time - start_time))
calcul_tx_erreur(X_dev, knn_res, Y_dev)


# --------------- Question 3 : PCA + classifieurs autres que DMIN ---------------

print(" -------------------- PCA + SVC --------------------\n")

# PCA faite en Q2, entrainement de SVC par rapport sur des donnees d'entrainement transformees par pca
obj_svc.fit(obj_pca.transform(X_trn), Y_trn)

# Estimation
start_time = time.time()
pca_svc_res = obj_svc.predict(obj_pca.transform(X_dev))
end_time = time.time()

print("Temps d'execution : %.2f sec" % (end_time - start_time))
calcul_tx_erreur(X_dev, pca_svc_res, Y_dev)


print(" -------------------- PCA + KNN --------------------\n")

# PCA faite en Q2, entrainement de neighbors par rapport sur des donnees d'entrainement transformees par pca
obj_knn.fit(obj_pca.transform(X_trn), Y_trn)

# Estimation
start_time = time.time()
pca_knn_res = obj_knn.predict(obj_pca.transform(X_dev))
end_time = time.time()

print("Temps d'execution : %.2f sec" % (end_time - start_time))
calcul_tx_erreur(X_dev, pca_knn_res, Y_dev)


# ----------------- Question 3 : Matrice de confusion -------------------
