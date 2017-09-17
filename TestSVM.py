import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

def ChooselinesToSpliteData(totalLines):
    percentToSave = 10

    # this func would return inedex list of lines to use for Learn, and lines to use for test


def main():

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
    #choose what lines to use for fitting and what lines for testing.

    linesForFitting =int(m-m*0.90)

    Y_Learn= Y[0:linesForFitting]
    X_Learn = X[0:linesForFitting,:]

    X_test = X[linesForFitting:numOflines,:]
    Y_test = Y[linesForFitting:numOflines]


    C = 1 # SVM regularization parameter
    degree = 2
    gamma  = 2
    svc =       svm.SVC(kernel='linear', C=C,decision_function_shape = 'ovr').fit(X_Learn,Y_Learn)

    print("C:={0} degree:={1} gamma:={2}".format(C,degree,gamma))
    #svc_predict
    svc_predict_result = np.equal(svc.predict(X_test),Y_test)
    svc_predict_score_test = svc_predict_result.sum()

    svc_predict_result = np.equal(svc.predict(X_Learn),Y_Learn)
    svc_predict_score_Learn = svc_predict_result.sum()


    tot_test = len(Y_test)
    print("correct predictions results on Test:")
    print('svc: {0} rbf_svc '.format(svc_predict_score_test))
    print('svc: {0:.2f}% '.format(
          svc_predict_score_test*100/tot_test))


    tot_Learn = len(Y_Learn)
    print("correct predictions results on Learn:")
    print('svc: {0} '.format(svc_predict_score_Learn))
    print('svc: {0:.2f}%'.format(
        svc_predict_score_Learn*100/tot_Learn))



if __name__ == "__main__":
    main()
