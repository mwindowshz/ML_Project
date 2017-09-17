import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# cost function
# (1/m)*sum((-y*log(h(x))-(1-y)*log(1-h(x))
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    #   r,c = X.shape
    #    XX = X.reshape(c,r)
    epsilon = 0.0001
    first = np.multiply(-y, np.log(sigmoid(np.dot(X, theta.T)) + epsilon))
    #  print(first)
    second = np.multiply((1 - y), np.log(1 - sigmoid(np.dot(X, theta.T)) + epsilon))
    # print(second)
    return np.sum(first - second) / (len(X))


# gradient_step,
def gradient_step(theta, X, y):
    lenX = len(X)

    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(np.dot(X, theta.T)) - y

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

#rowsToRead = 0
#Y = data.iloc[rowsToRead:,0:1]
##convert to 0/1 malignat
#Y = pd.get_dummies(Y,drop_first='M')
#X = data.iloc[rowsToRead:,1:n]



rowsToRead = 0
# we need to shuffle the data because it is in order of
shuffeldData = data.sample(frac=1).reset_index(drop=True)

# collect features v1 to v20
X = shuffeldData.iloc[:, 0:n-1]
Y = shuffeldData.iloc[:, n-1:n]
#convert to 0/1 malignat
Y = pd.get_dummies(Y,drop_first='M')
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
linesToUseForLearn = 300
x_Learn = X[0:linesToUseForLearn, :]
y_Learn = y[0:linesToUseForLearn]

X_test = X[linesToUseForLearn+1:m, :]
Y_test = y[linesToUseForLearn+1:m]


# find best features:
OptimalFeaturesData = []
feature_from_X_to_Test = []
costForEachTestedRound = np.zeros(n)
numOfIterations = 50
chosenFeaturesList = []
alpha = 0.3
# initialize - first we use only 1 feature, then we would gradually increase
numOfFeatureToUse = 1

curr_feature = 0
np.ones(linesToUseForLearn)

for times in range(5):

    curr_feature = 0
    for curr_feature in range(n):
        featuresToLearnOn = np.array(np.ones(linesToUseForLearn))
        # featuresToLearnOn.append(np.ones(linesToUseForLearn))
        if times == 0:
            featuresToLearnOn = np.c_[featuresToLearnOn, x_Learn[:, curr_feature]]
        else:
            for i in range(len(chosenFeaturesList)):
                featuresToLearnOn = np.c_[featuresToLearnOn, x_Learn[:, chosenFeaturesList[i]]]
                # featuresToLearnOn.append(x_Learn[:,chosenFeaturesList[i]].T)
            # add the new feature only if it was not on the chosenFeaturesList
            inArray = 0
            for i in range(len(chosenFeaturesList)):
                if curr_feature == chosenFeaturesList[i]:
                    inArray = 1
            if inArray == 0:
                featuresToLearnOn = np.c_[featuresToLearnOn, x_Learn[:, curr_feature]]
        r, c = np.array(featuresToLearnOn).shape
        theta = np.zeros(c)
        for i in range(numOfIterations):
            for j in range(c):
                theta[j] = theta[j] - alpha * gradient_step(theta, featuresToLearnOn, y_Learn)[j]

        costForEachTestedRound[curr_feature] = cost(theta, featuresToLearnOn, y_Learn)
        # curr_feature+=1

    feature_index_to_add_to_list = costForEachTestedRound.argmin()
    chosenFeaturesList.append(costForEachTestedRound.argmin())
    # ---make a list only of indexes and in loop use these indexes to construct the list of x's to use'
    OptimalFeaturesData.append(x_Learn[feature_index_to_add_to_list])
    numOfFeatureToUse += 1

# after we found the 5 best features, now lets use them on our data and see the results of the predictions
print('Best 5 features are {0} Iterations:{1} alpha{2}'.format(chosenFeaturesList, numOfIterations, alpha))

# create X ,Y and Theta values list for final prediction and data for testing
mini_X_Learn_List = featuresToLearnOn  # np.array(np.ones(linesToUseForLearn))
mini_Y_LearnList = np.array(np.ones(1))
Y_test.size
testSize = Y_test.size

mini_X_Test = np.array(np.ones(testSize))

r, c = np.array(featuresToLearnOn).shape
theta = np.zeros(c)
for i in chosenFeaturesList:
    mini_X_Test = np.c_[mini_X_Test, X_test[:, i]]
    mini_Y_LearnList = np.append(mini_Y_LearnList, y_Learn[i])

costForEachTestedRound = np.zeros(n)
# numOfIterations = 100
initial_cost = cost(theta, mini_X_Learn_List, mini_Y_LearnList)
alpha = 0.3

minCost = initial_cost
current_cost = minCost
passedMinimum = 0

cos_plot_vals = [None] * numOfIterations

numOfFeaturesUsed = 5 + 1

# we loop in numOfIterations, in order to reduce the cost, and get to better set of theta parameters
for i in range(numOfIterations):
    # perform the gradient decent, n is number of features in X so e compute theta_j=thetaJ -alpha*g(theta.T*X)
    # and update all the theta variables
    for j in range(numOfFeaturesUsed):
        theta[j] = theta[j] - alpha * gradient_step(theta, mini_X_Learn_List, y_Learn)[j]
    current_cost = cost(theta, mini_X_Learn_List, y_Learn)
    cos_plot_vals[i] = current_cost
# display the cost lowring during the gradient decent:
plt.plot(cos_plot_vals)
plt.show()


# test prediction
theta_min = theta

predictions = predict(theta_min, mini_X_Test)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y_test)]
accuracy = (sum(map(int, correct)) / len(Y_test))
print('accuracy on test data = {0}%'.format(accuracy))

# test prediction
theta_min = theta
# theta_min = np.matrix(theta])
predictions = predict(theta_min, mini_X_Learn_List)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y_Learn)]
accuracy = (sum(map(int, correct)) / len(y_Learn))
print('accuracy on learning data= {0}%'.format(accuracy))


