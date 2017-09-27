import DataLoader
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.externals import joblib
from time import time
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)

    end = time()
    #find on lines prediction faild
    y_pred2 = pd.get_dummies(y_pred,drop_first='M')
    y_orig2 = pd.get_dummies(target,drop_first='M')
    diff = np.array(y_pred2-y_orig2)
    indexes =[]
    for i,j in enumerate(diff):
        if j[0]!=0.0:
            indexes.append(i)
    print("Total number of wrong predictions are: {} from {}.\nWrong line indexes: {}\n".format(len(indexes),len(target),indexes))
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred, pos_label='M'), sum(target == y_pred) / float(len(y_pred))
def RunSavedClassifier(savedClassifilerPKL,X,Y):
    try:
        estimator = joblib.load(savedClassifilerPKL)
        print("using trained model")
    except:
        print("building new model")
    estimator.fit(X, Y.values.ravel())


    # Report the final F1 score for training and testing after parameter tuning
    f1, acc = predict_labels(estimator, X, Y.values.ravel())
    print("Estimator F1 score and Accuracy score for load data set: {:.4f} , {:.4f}.".format(f1, acc))





print("loading Data file:{}".format(sys.argv[1]))

filename = sys.argv[1]
# filename = "Data/real_project_data.xls"

Data = DataLoader.LoadData(filename)
m,n = Data.shape
X_all = Data.iloc[:, 0:n-1]
y_all = Data.iloc[:, n-1:n]
#change Y with get_dummies to 0,1
#y_all = pd.get_dummies(y_all,drop_first='M')

#normalize Data
X_all = preprocessing.scale(X_all)

RunSavedClassifier("SVC_Params.pkl",X_all,y_all)

# divider = 10
# X_train, X_test, y_train, y_test = DataLoader.split_data(Data,True,divider,2)
# y_train = pd.DataFrame(y_train)
# y_test = pd.DataFrame(y_test)
# RunSavedClassifier("SVC_Params.pkl",X_train,y_train)
# RunSavedClassifier("SVC_Params.pkl",X_test,y_test)