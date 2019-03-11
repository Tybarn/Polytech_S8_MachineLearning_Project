import numpy as np

from functions_dmin import dmin
from functions_pca import pca
from functions_sklearn import svc

def calcul_tx_erreur(X, res_obt, res_waited):
    diff_res = np.array(res_obt == res_waited)
    print("Taux d'erreur SVC: ")
    print(np.array(X[diff_res == False]).shape[0] / X.shape[0] * 100)

X_trn = np.load('../data/trn_img.npy')
Y_trn = np.load('../data/trn_lbl.npy')

X_dev = np.load('../data/dev_img.npy')
Y_dev = np.load('../data/dev_lbl.npy')

X_tst = np.load('../data/tst_img.npy')

res_pca = pca(X_trn, X_dev, 182) # limite à 182
res_dmin = dmin(res_pca, Y_dev)
calcul_tx_erreur(X_dev, res_dmin, Y_dev)

#res_svc = svc(X_trn, Y_trn, X_dev)