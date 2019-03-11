from sklearn import svm

def svc(fit_data, fit_res, predict_data):
    clf = svm.SVC(gamma='scale')
    clf.fit(fit_data, fit_res)
    return clf.predict(predict_data)