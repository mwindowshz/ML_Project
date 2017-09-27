
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
# Standardising the data.
from sklearn.preprocessing import scale
from pandas.tools.plotting import scatter_matrix

from time import time
from sklearn.metrics import f1_score
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
#import xgboost as xgb
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
from sklearn import svm, datasets
#broot force search best parameters
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
#save model to file
from sklearn.externals import joblib



def LoadData(filename):
    # read input data

    data = pd.read_excel(filename)
    # m- num of samples, n-num of colums
    # note  - number first colume is the y values, so we would split data into Y vector and X matrix
    [m, n] = data.shape
    return data

def split_data(data, randomizeData, divider, normalizeType):

    # scatter_matrix(data,alpha=0.2, figsize=(6, 6), diagonal='kde')

    [m, n] = data.shape
    if randomizeData:
        shuffeldData = data.sample(frac=1).reset_index(drop=True)
    else:
        shuffeldData = data
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

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print( "Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)

    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred, pos_label='M'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print( f1, acc)
    print( "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print(  "F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))

def RunSavedClassifier(savedClassifilerPKL,XTrain,yTrain,XTest,YTest):
    try:
        estimator = joblib.load(savedClassifilerPKL)
        print("using trained model")
    except:
        print("building new model")
    estimator.fit(XTrain, yTrain.values.ravel())
    #joblib.dump(estimator,"/my_models/%s.pkl"%dataset_name)

    # Report the final F1 score for training and testing after parameter tuning
    f1, acc = predict_labels(estimator, XTrain, yTrain.values.ravel())
    print("estimator F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(estimator, XTest, YTest.values.ravel())
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


#main program area


filename = "Data/real_project_data.xls"
data = LoadData(filename)

#scatter_matrix(data,alpha=0.2, figsize=(6, 6), diagonal='kde')

[m,n] = data.shape
shuffeldData = data.sample(frac=1).reset_index(drop=True)

# collect features v1 to v20
X_all = shuffeldData.iloc[:, 0:n-1]
y_all = shuffeldData.iloc[:, n-1:n]
#change Y with get_dummies to 0,1
#y_all = pd.get_dummies(y_all,drop_first='M')

#normalize Data
X_all = preprocessing.scale(X_all)
#scale(X_all)
# normalize X
#xMean = np.mean(X_all, axis=0)
#xStd = np.std(X_all, axis=0)
#xStd += 0.00001
#h = (X_all - xMean) / xStd

divider = 5
# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size = int(m/divider),
                                                    random_state = 2,
                                                    stratify = y_all)



# RunSavedClassifier('SVC_Params.pkl',X_train,y_train,X_test,y_test)

#create classifiers

# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state = 42, verbose=True)
clf_B = SVC(random_state = 912, kernel='rbf')
#Boosting refers to this general problem of producing a very accurate prediction rule
#by combining rough and moderately inaccurate rules-of-thumb
#clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, X_train, y_train.values.ravel(), X_test, y_test.values.ravel())
print ('')
train_predict(clf_B, X_train, y_train.values.ravel(), X_test, y_test.values.ravel())
print ('')
#train_predict(clf_C, X_train, y_train, X_test, y_test)
print ('')




parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train,y_train.values.ravel())
#broot force find best hyper parameters
# TODO: Create the parameters list you wish to tune
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf']},
  #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree': [2, 3, 4, 5, 8], 'kernel': ['poly']},
  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']}
 ]


clf = SVC() #random_state = 912, kernel='rbf')

# TODO: Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label='M')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=param_grid,
                        cv=5)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train.values.ravel())

# Get the estimator
clf = grid_obj.best_estimator_
print("")


# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train.values.ravel())
print("GridSearchCV F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

f1, acc = predict_labels(clf, X_test, y_test.values.ravel())
print("GridSearchCV F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))

#save Classifier with best parames
joblib.dump(clf, "SVC_Params.pkl")
