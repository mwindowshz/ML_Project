# Project Report

## Used Algorithms

* Logistic regression 
* Logistic with best N features
* SVM with different kernels and parameters
* Neural Network

## Tools

* PyCharm
* NumPy
* SkLearn
* Matplotlib
* Keras
* Own implementations

## Finding Best Algorithm

In order to find the best classifier for the given dataset we used several approaches and selected the best performing one. 

**Normalization**

**Randomization**

### Logistic Regression (*LR*)

#### SK Learn
* Run with different parameters

Although *LR* seems to be a good choice, the different parameters did not always bring different results. The overall success rate was between 90-92 %.

# TODO: Add image ...logisticRegressionSklearn

![Image of Yaktocat](images/logistic_sklearn_res.PNG)



#### Self Implemented LR


#### Finding X best features (LogisticRegression2.py)
To narrow down and reduce the complexity of the dataset we tried to single out the best feature, i.e. features which let us learn best from the data and ignore the majority of other not significant ones. The code for *LR* used here is our own implementation.

**Approach**

```
best_features = []

for feature in not_yet_selected_features
  learn on [feature, ...best_features]
  
  
  
best_features.push(next_best_feature)
 
```
* Run *LR* with only one feature 
* Select the best performing one and add it to selected_features list
* Repeat adding one feature at a time together with already selected and selecting next best performing feature
* Run *LR* on the whole dataset, but only on X selected features

# TODO: Add images:
* LogisticRegersion_On_X_Best_restExample.png
* LogisticRegersion_On_X_Best.png


**Conclusion**
Running the feature selection multiple times with randomized train/test split we noticed that certain features were always included in the 'winner' set, whereas others we only sometimes included. It seems that the features which are always present are really representative for the whole dataset whereas other are less representative or significant in general, or the differences between them are tinier.

### SVM (RunClassifier.py)

* Added SVM with "rbf" kernel
* Got good results (93%) without any changes

This brought us to the conclusion that *SVM* could be the best fitting classifier for our dataset. Thus we wanted to optimize it even further and finetune the following parameters:

* kernel
* regularization
* other kernel specific parameters

```
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree': [2, 3, 4, 5, 8], 'kernel': ['poly']},
  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']}
 ]
```

SKLearn provides an important and useful feature called *GridSearchCV* which allows us to easily define an array of parameters and autoruns all the models, returning the best performing ones. As seen in the code sample, ` {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']}`, the `sigmoid` kernel was run with 5 different regularization paremeters:  [0.1, 1, 10, 100, 1000], and 2 different gamma:  [0.001, 0.0001].

# TODO: add image : gridSearch_res

The *GridSearchCV* function allows us to save the best performing model including the parameters in order to be able to run it on other data without running the comparisons again. `joblib.dump(clf, "SVC_Params.pkl")`

## Neural Network (NeuralNetwork.py)



We used the "Keras" library with the Theano backend. We tried to build several simple network architectures. The final one presented here looks as follows:

* First layer is the input (size: 20)
* Fully connected layer (size: 20)
* Dropout layer (non learning layer, has no parameters)
* FC Layer (size: 64) 
* Dropout
* Classifier layer (with a sigmoid activation)

We used different optimizers (sgd, rmsprop, adagrad, adam) with different specific hyperparameters.

In all the scenarios we were only able to achieve results not as good as the ones we've got using *SVM* or *LR*. There are different reasons which in our view could explain this anomaly:

* The network architecture does not fit the dataset well enough
* The training dataset is too small
* Wrong paremters selected due to time constraints and only partial knowledge of the approach

Out of all methods the Neural Network urged us to explore it in more depth in future.

# TODO: add images: 
3 images with neural networks


## Conclusion

We saved the model as described in the *SVM* section. 
