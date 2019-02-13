import numpy as np

from functions_dmin import dmin

X = np.load('../data/trn_img.npy')
Y = np.load('../data/trn_lbl.npy')

dmin(X,Y)