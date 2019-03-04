from sklearn.decomposition import PCA

def pca(fit_data, data, nb_components):
    obj_pca = PCA(n_components=nb_components, svd_solver='full')
    obj_pca.fit(fit_data)
    new_data = obj_pca.transform(data)
    return new_data