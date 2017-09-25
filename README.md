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

### Logistic Regression (*LR*)

#### SK Learn


#### Self Implemented


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

