import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from sklearn.cross_validation import train_test_split

'''in this excersize we would need to find the 5 most influential features in our dataset, 
we would use the following stratigy
1. run the logistic regression, using only one feature, and save the cost value , do this for each of the features.
2. choose the feature were the cost is minimal.
3. run again using the chosen feature, and add another one. save cost over all tries,add the minimal costing feature to our best feature list
'''


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# cost funciton
# (1/m)*sum((-y*log(h(x))-(1-y)*log(1-h(x))
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    epsilon = 0.0001
    first = np.multiply(-y, np.log(sigmoid(X * theta.T) + epsilon))
    #  print(first)
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T) + epsilon))
    # print(second)
    return np.sum(first - second) / (len(X))


# gradient_step,
def gradient_step(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    z = X * theta.T
    np.exp(-z)
    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    # print(grad)
    return grad


# predict, use Theata and run the h(x) hipothesis and decide for each line/set of features, what we predict using h(x)  in this case h(x) is sigmoid func
def predict(theta, X):
    probability = sigmoid(X.dot(theta.T))
    # return [1 if p >= 0.5 else 0 for p in probability]
    res = []
    for p in probability:
        if p >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return res


# program main :

# read input data
filename = "C:\\Users\\owner\\Google Drive\\docs\\study\\ML_final_Project\\real_project_data.xls"
data = pd.read_excel(filename)
# m- num of samples, n-num of colums
# note  - number first colume is the y values, so we would split data into Y vector and X matrix
[m, n] = data.shape
rowsToRead = 0
# we need to shuffle the data because it is in order of
shuffeldData = data.sample(frac=1).reset_index(drop=True)

# collect features v1 to v20
X = shuffeldData.iloc[:, 0:19]
Y = shuffeldData.iloc[:, 20]

# set size of X to be size of theta later so we add a whole colume of 'ones'
X.insert(0, 1, 1)
# convert to numpy matix and array
# XX = X.values
# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(Y.values)

theta = np.zeros(n)
# normalize X
xMean = np.mean(X, axis=0)
xStd = np.std(X, axis=0)
xStd += 0.00001
X = (X - xMean) / xStd
X.shape, theta.shape, y.shape

# at this point we have  all the data normalized, now we would create the set for learning and set for testing

x_Learn = X[0:300, :]
y_Learn = y[0:300]

X_test = X[301:m, :]
Y_test = y[301:m]

# test
# find best features:
OptimalFeaturesList = []
feature_from_X_to_Test = []
costForEachTestedRound = np.zeros(300)
numOfIterations = 100

alpha = 0.3
numOfFeatureToUse = 1
theta = np.zeros(n)

curr_feature = 0
for x in x_Learn - (numOfFeatureToUse - 1):
    feature_from_X_to_Test = x
    for i in range(numOfIterations):
        for j in range(n):
            theta[j] = theta[j] - alpha * gradient_step(theta, feature_from_X_to_Test, y_Learn)[j]

    costForEachTestedRound[curr_feature] = cost(theta, x_Learn, y_Learn)
    curr_feature += 1

feature_index_to_add_to_list = costForEachTestedRound.argmin()
OptimalFeaturesList.append(x_Learn[feature_index_to_add_to_list])
numOfFeatureToUse += 1
