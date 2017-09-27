# Project 

# TODO: add names and ids and Github repo




## Finding Best Algorithm

In order to find the best classifier for the given dataset we used several approaches and selected the best performing one. In the following report we will compare:

* Logistic Regression 
* SVM
* Neural Network


## Data Preprocessing

### Normalization

Following best practices learned during the course, we normalized the given dataset before applying any of the classification techniques. The normalization was partly done by hand using mean and standard deviation, and partly using the build in sklearn preprocessing feature with `scale` or `normalize` funcitons.

### Dealing with Overfitting

In all scenarios we split the data in to train/test groups using the sklearn `train_test_split` function. We used the randomization parameter, such that each and every run had a unique train/test set. 

Additionally, we tried changing the split ratio between train/test, i.e. 90% train and 10% test, and others.


## Classifiers

### Logistic Regression (*LR*)

#### Using SKLearn

In the experiments we run, different parameters were used. Although *LR* seems to be a good choice, the different parameters did not always bring different results. The overall success rate was between 90-92 %.

![](images/logistic_sklearn_res.PNG)


#### Using Self Implemented *LR*
##### Finding X best features
To narrow down and reduce the complexity of the dataset we tried to single out the best features, i.e. features which let us learn best from the data and ignore the majority of other not significant ones. The code for *LR* used here is our own implementation and can be found in `LogisticRegression2.py`

##### Approach

* Run *LR* with only one feature 
* Select the best performing one and add it to selected_features list
* Repeat adding one feature at a time together with already selected and selecting next best performing feature
* Run *LR* on the whole dataset, but only on X selected best features


![](images/LogisticRegersion_On_X_Best_restExample.png)
###### Fig: 5 Best features


##### Conclusion
Running the feature selection multiple times with randomized train/test split we noticed that certain features were always included in the 'winner' set, whereas others we only sometimes included. It seems that the features which are always present are really representative for the whole dataset whereas other are less representative or significant in general, or the differences between them are tinier.

##### Hyperparameters for *LR*

Out of different performed executions with different hyperparameters like `alpha`, `# of iterations`, here are a few examples of the results.



![](images/LogisticRegersion_On_X_Best.png)
###### Fig: (Cost of *LR*: `alpha = 0.3`, `# of iterations = 50`, `accuracy = ~92%`)


![](images/logistic_0.03_100iter.png)
###### Fig: (Cost of *LR*: `alpha = 0.03`, `# of iterations = 100`, `accuracy = ~89%`)


As we can see, by lowering the alpha from 0.3 to 0.03 and keeping the number of iterations fixed, the model delivers far worse results. Although we tried to increase the iteration size to even 500, it was able to reduce the costs of the gradient descent only slightly. Thus finding the appropriate alpha seemed to be a more reasonable approach. 



### SVM

We used *SKLearn SVM* in out of the box with the default `rbf` kernel and observed good results of accuracy `~93%` without further tweaking. (The file can be found in `RunClassifier.py` in the GitHub repo.)

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

SKLearn provides an important and useful feature called *GridSearchCV* which allows us to easily define an array of parameters and autoruns all the models, returning the best performing one. As seen in the code sample, ` {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']}`, the `sigmoid` kernel was run with 5 different regularization paremeters:  [0.1, 1, 10, 100, 1000], and 2 different gammas:  [0.001, 0.0001].


![](images/gridSearch_res.PNG)
###### Fig: One of the *GridSearchCV* results


The *GridSearchCV* function allows us to save the best performing model including the parameters in order to be able to run it on other data without running the comparisons again. It can be done using the command: `joblib.dump(clf, "SVC_Params.pkl")`.

In the final `test.py` file we made use of this feature.

## Neural Network (NeuralNetwork.py)

Although *SVM* brought good results, we wanted to explore other options and experimented with *Neural Networks (NN)*

We used the "Keras" library with the Theano backend. We tried to build several simple network architectures. 

![](images/Net_1.png)
###### Fig: Network - Input20, FC20, FC8, FC1 (Classification)


The one presented here looks as follows:

* First layer is the input (size: 20)
* Fully connected layer (size: 20)
* Dropout layer (non learning layer, has no parameters)
* FC Layer (size: 64) 
* Dropout
* Classifier layer (with a sigmoid activation)


![](images/Net_2.png)
###### Fig: Other Network Example


### Exploring Model Loss

In the following 2 figures we see two graphs of the model loss. The first is more smooth while the other one shows more jitter. This phenomenon is due to the dropout layers of the network. We explored adding dropout layers in hope to get a more robust classifier only to find that the total accuracy and loss were barely changed.


![](images/net_1_loss.png)
###### Fig: Network 1 loss


![](images/Net_modelLoss.png)
###### Fig: Network 2 loss



We used different optimizers (sgd, rmsprop, adagrad, adam) with different specific hyperparameters.

In all the scenarios we were only able to achieve results not as good as the ones we've got using *SVM* or *LR*. There are different reasons which in our view could explain this anomaly:

* The network architecture does not fit the dataset well enough
* The training dataset is too small
* Wrong paremters selected due to time constraints and only partial knowledge of the approach

Out of all methods the Neural Network urged us to explore it in more depth in future.

![](images/Net_Learning_Progress.png)


## Running the Final Test File

We saved the model as described in the *SVM* section, in the `Deliverables` folder in the GitHub repo. Steps to order to run the program:

```
* unzip the archive
* navigate to the folder in the terminal
* run: python test.py full_path_to_your_test_data_set
* example: python test.py C:\Users\dataset.xls

```

## Appendix

### Tools used

* PyCharm
* NumPy
* SkLearn
* Matplotlib
* Keras
* Own implementations
