import numpy as np

from functions_dmin import dmin
from functions_pca import pca

X_trn = np.load('../data/trn_img.npy')
Y_trn = np.load('../data/trn_lbl.npy')

X_dev = np.load('../data/dev_img.npy')
Y_dev = np.load('../data/dev_lbl.npy')

X_tst = np.load('../data/tst_img.npy')

data = pca(X_trn, X_dev, 182) # limite à 182
dmin(data,Y_dev)