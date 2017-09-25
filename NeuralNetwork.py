# from keras.models import Sequential
import DataLoader
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import initializers
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from keras.optimizers import SGD, RMSprop

filename = "Data/real_project_data.csv"

# Data =DataLoader.LoadData(filename)
# X = Data.iloc[:, 0:20].astype(float)
# Y = Data.iloc[:,20]

# filename = "C:\\Users\\owner\\Documents\\dataset.csv"

# load pima indians dataset
dataset = np.loadtxt(filename, delimiter=',')

# split into input and output variables
X = dataset[:, 0:20]
Y = dataset[:, 20]

# normalize X
X = preprocessing.normalize(X)
seed = 9
np.random.seed(seed)
# split the data into training (67%) and testing (33%)
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

#
kernelInit = initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None)
model = Sequential()
model.add(Dense(20, input_dim=20, activation='relu',    kernel_initializer='he_normal',bias_initializer='zeros'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              # optimizer='rmsprop',
              # optimizer=sgd,
              # optimizer=rms,
              # optimizer=RMSprop(lr=0.1),
              optimizer='adam',
              # optimizer='adagrad',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
          epochs=500,
          batch_size=40, verbose=1)
score = model.evaluate(x_test, y_test, batch_size=50)
print("\nAccuracy: %.2f%%" % (score[1] * 100))
model.summary()


# # list all data in history
# print(history.history.keys())
# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# # confusion train
# y_true = np.argmax(y_train, axis=0)
# y_pred = np.argmax(model.predict(x_train), axis=1)
# print(confusion_matrix(y_true, y_pred))
# #
# # confusion test
# y_true = np.argmax(y_test, axis=1)
# y_pred = np.argmax(model.predict(x_test), axis=1)
# print(confusion_matrix(y_true, y_pred))
#
# score = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# 87%
# model = Sequential()
# model.add(Dense(20, input_dim=20, kernel_initializer='uniform', activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(10, input_dim=20, kernel_initializer ='uniform', activation='relu'))
# model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# # compile the model
# # model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # fit the model
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=5, verbose=1)
#
# # evaluate the model
# scores = model.evaluate(x_test, y_test)
# print("Accuracy: %.2f%%" % (scores[1] * 100))
#
# m_numOfLines, numberOfFeatures = np.shape(x_train)
# print(m_numOfLines)

#
# from keras.layers import Input, Dense
# from keras.models import Model
# from keras.utils import np_utils
# # This returns a tensor
# inputs = Input(shape=(20,))
#
# # a layer instance is callable on a tensor, and returns a tensor
# x = Dense(64, activation='tanh')(inputs)
# x = Dense(64, activation='tanh')(x)
# predictions = Dense(1, activation='softmax')(x)
#
# # This creates a model that includes
# # the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=255, batch_size=5, verbose=2)
#
# # evaluate the model
# scores = model.evaluate(X_test, Y_test)
# print ("Accuracy: %.2f%%" %(scores[1]*100))
#
# # [m, n] = Data.shape
# # #shuffeldData = data.sample(frac=1).reset_index(drop=True)
# #
# # # collect features v1 to v20
# # # split into input (X) and output (Y) variables
# # X = Data.iloc[:, 0:20].astype(float)
# Y = Data.iloc[:,20]
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# # from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# # from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# # from sklearn.model_selection import StratifiedKFold
# #def create_baseline(n):
# 	# create model
# model = Sequential()
# model.add(Dense(20, input_dim=20, kernel_initializer='normal', activation='relu'))
# model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #	return model
#
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# # evaluate model with standardized dataset
# # model = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
# model.fit(X,encoded_Y)
#
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# #
# # model.fit(X_all, newY,batch_size=10,epochs=50)  # starts training
#
#
#
#
# #
# # model = Sequential()
# #
# # from keras.layers import Dense, Activation
# #
# # model.add(Dense(units=64, input_dim=100))
# # model.add(Activation('relu'))
# # model.add(Dense(units=10))
# # model.add(Activation('softmax'))
# #
# # model.compile(loss='categorical_crossentropy',
# #               optimizer='sgd',
# #               metrics=['accuracy'])
# #
# #
# # maybe switch to functional api
# #
# # from keras.models import Sequential
# #
# #
# # model = Sequential()
# #
# # from keras.layers import Dense, Activation
# #
# # model.add(Dense(units=64, input_dim=100))
# # model.add(Activation('relu'))
# # model.add(Dense(units=10))
# # model.add(Activation('softmax'))
# #
# # model.compile(loss='categorical_crossentropy',
# #               optimizer='sgd',
# #               metrics=['accuracy'])
# # # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
# # model.fit(X_train, y_train, epochs=5, batch_size=32)
# #
# # loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
# #
# # classes = model.predict(X_test, batch_size=128)
