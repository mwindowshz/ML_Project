import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm


filename = "C:\\Users\\owner\\Google Drive\\docs\\study\\ML_final_Project\\real_project_data.xls"
data = pd.read_excel(filename)
[m,n] = data.shape

print(m)
print(n)

rowsToRead = 0
#we need to shuffle the data because it is in order of
shuffeldData = data.sample(frac=1).reset_index(drop=True)

#collect features v1 to v20
X = np.array(shuffeldData.iloc[:,0:19])
Y = np.array(shuffeldData.iloc[:,20])

numOflines = len(Y)
linesForFitting =int(m-m*0.25)
Y_Learn= Y[0:linesForFitting]
X_Learn = X[0:linesForFitting,:]

X_test = X[linesForFitting:numOflines,:]
Y_test = Y[linesForFitting:numOflines]





clf = svm.SVC(gamma=0.01,C=100)
##read data in x , and classification in y

clf.fit(X_Learn,Y_Learn)



# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1 # SVM regularization parameter
degree = 2
gamma  = 2
svc =       svm.SVC(kernel='linear', C=C,decision_function_shape = 'ovr').fit(X_Learn,Y_Learn)
rbf_svc =   svm.SVC(kernel='rbf', C=C,gamma=gamma).fit(X_Learn,Y_Learn)
poly_svc =  svm.SVC(kernel='poly', degree=degree, C=C,gamma=gamma).fit(X_Learn,Y_Learn)
sigmoid_svc = svm.SVC(kernel='sigoid',C=C,gamma = gamma,coef0=999)
lin_svc = svm.LinearSVC(C=C).fit(X_Learn,Y_Learn)
print("C:={0} degree:={1} gamma:={2}".format(C,degree,gamma))
#svc_predict
svc_predict_result = np.equal(svc.predict(X_test),Y_test)
svc_predict_score_test = svc_predict_result.sum()

svc_predict_result = np.equal(svc.predict(X_Learn),Y_Learn)
svc_predict_score_Learn = svc_predict_result.sum()

#rbf_svc_predict
rbf_svc_predict_result = np.equal(rbf_svc.predict(X_test),Y_test)
rbf_svc_predict_score_test = rbf_svc_predict_result.sum()

rbf_svc_predict_result = np.equal(rbf_svc.predict(X_Learn),Y_Learn)
rbf_svc_predict_score_Learn = rbf_svc_predict_result.sum()

#poly_svc_predict
poly_svc_predict_result = np.equal(poly_svc.predict(X_test),Y_test)
poly_svc_predict_score_test = poly_svc_predict_result.sum()

poly_svc_predict_result = np.equal(poly_svc.predict(X_Learn),Y_Learn)
poly_svc_predict_score_Learn = poly_svc_predict_result.sum()

#lin_svc_predict
lin_svc_predict_result = np.equal(lin_svc.predict(X_test),Y_test)
lin_svc_predict_score_test = lin_svc_predict_result.sum()

lin_svc_predict_result = np.equal(lin_svc.predict(X_Learn),Y_Learn)
lin_svc_predict_score_Learn = lin_svc_predict_result.sum()


#clf_predict
clf_predict_result = np.equal(clf.predict(X_test),Y_test)
clf_predict_score = clf_predict_result.sum()

clf_predict_result = np.equal(clf.predict(X_Learn),Y_Learn)
clf_predict_score_Learn = clf_predict_result.sum()

#sigmoid_predict
sigoind_predict_result = np.equal(clf.predict(X_test),Y_test)
sigmoid_predict_score = sigoind_predict_result.sum()

sigoind_predict_result = np.equal(clf.predict(X_Learn),Y_Learn)
sigmoid_predict_score_Learn = sigoind_predict_result.sum()

tot_test = len(Y_test)
print("correct predictions results on Test:")
print('svc: {0} rbf_svc {1} poly_svc {2} lin_svc {3} clf {4} sigmoid_svc {5}'.format(svc_predict_score_test,rbf_svc_predict_score_test,poly_svc_predict_score_test,lin_svc_predict_score_test,clf_predict_score,sigmoid_predict_score))
print('svc: {0:.2f}% rbf_svc {1:.2f}% poly_svc {2:.2f}% lin_svc {3:.2f}% clf {4:.2f}% sigoind_svc:{5:.2f}'.format(
      svc_predict_score_test*100/tot_test,
      rbf_svc_predict_score_test*100/tot_test,
      poly_svc_predict_score_test*100/tot_test,
      lin_svc_predict_score_test*100/tot_test,
      clf_predict_score*100/tot_test,
      sigmoid_predict_score*100/tot_test))

tot_Learn = len(Y_Learn)
print("correct predictions results on Learn:")
print('svc: {0} rbf_svc {1} poly_svc {2} lin_svc {3} clf {4} sigmoid_svc {5}'.format(svc_predict_score_Learn,rbf_svc_predict_score_Learn,poly_svc_predict_score_Learn,lin_svc_predict_score_Learn,clf_predict_score_Learn,sigmoid_predict_score_Learn))
print('svc: {0:.2f}% rbf_svc {1:.2f}% poly_svc {2:.2f}% lin_svc {3:.2f}% clf {4:.2f}% sigoind_svc:{5:.2f}'.format(
      svc_predict_score_Learn*100/tot_Learn,
      rbf_svc_predict_score_Learn*100/tot_Learn,
      poly_svc_predict_score_Learn*100/tot_Learn,
      lin_svc_predict_score_Learn*100/tot_Learn,
      clf_predict_score_Learn*100/tot_Learn,
      sigmoid_predict_score_Learn*100/tot_Learn))


