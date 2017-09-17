import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing



def LoadData(filename):
    # read input data

    data = pd.read_excel(filename)
    # m- num of samples, n-num of colums
    # note  - number first colume is the y values, so we would split data into Y vector and X matrix
    [m, n] = data.shape
    return data

# this funciton recieves the data, and splits it in to train and test x,y The funciton also normalizes the data
def split_data(data, randomizeData, divider, normalizeType):

    # scatter_matrix(data,alpha=0.2, figsize=(6, 6), diagonal='kde')

    [m, n] = data.shape
    shuffeldData = data.sample(frac=1).reset_index(drop=True)

    # collect features v1 to v20
    X_all = shuffeldData.iloc[:, 0:n - 1]
    y_all = shuffeldData.iloc[:, n - 1:n]
    # change Y with get_dummies to 0,1
    # y_all = pd.get_dummies(y_all,drop_first='M')

    # h = preprocessing.scale(X_all)
    # scale(X_all)

    # normalize X
    if (normalizeType == 1):
        xMean = np.mean(X_all, axis=0)
        xStd = np.std(X_all, axis=0)
        xStd += 0.00001
        X_all = (X_all - xMean) / xStd
    if (normalizeType == 2):
        X_all = preprocessing.scale(X_all)
    if (normalizeType == 3):
        X_all = preprocessing.normalize(X_all)

    # Shuffle and split the dataset into training and testing set.
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                        test_size=int(m / divider),
                                                        random_state=2,
                                                        stratify=y_all)

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
