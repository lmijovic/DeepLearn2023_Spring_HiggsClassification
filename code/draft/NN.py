# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:08:50 2021

@author: keira
"""

import numpy as np
import matplotlib.pyplot as plt

from atlasify import atlasify

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from keras import callbacks

import plots


np.random.seed(1)

# Number of inputs
n = 25

# Number of epochs
nep = 200

# Import data
data = np.loadtxt('features.csv', delimiter = ',')
X = data[:,:n]
y = data[:,n]

# Feature normalization
normalize = StandardScaler()
X_fit = normalize.fit_transform(X)

# Split data into training & testing samples 
X_train, X_test, y_train, y_test = train_test_split(X_fit, y, test_size = 0.2, random_state = 1)

# Build NN
model = Sequential()

model.add(Dense(24, input_dim=n, activation = 'relu', activity_regularizer = l2(1e-4)))
model.add(Dense(16, activation = 'relu', activity_regularizer = l2(1e-4)))
model.add(Dense(8, activation = 'relu', activity_regularizer = l2(1e-4)))
model.add(Dense(1, activation = 'sigmoid', activity_regularizer = l2(1e-4)))

plot_model(model, to_file='NN-architecture.png', show_shapes=True, show_layer_names=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

stop = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = True)

# Training process
hist = model.fit(X_train, y_train, epochs=nep, batch_size=5000, validation_split = 0.2, callbacks = [stop])
    
# Loss plots
loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
plt.plot(range(1,len(loss_train)+1), loss_train, 'g', label='Training loss')
plt.plot(range(1,len(loss_val)+1), loss_val, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
atlasify('work in progress')
plt.legend(frameon = False)
plt.show()

# ROC curve
y_pred = model.predict(X_test).flatten()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
AUC = auc(fpr, tpr)
plt.plot(fpr,tpr, 'k-')
plt.plot([0,1],[0,1],'r--')
plt.text(0.8,0, 'AUC = ' + str(np.round(AUC, 5)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
atlasify('work in progress', enlarge = 1, outside = True)
plt.show()

# Write outfile
myy = normalize.inverse_transform(X_test)[:,0]
res = np.array([myy.T, y_pred, y_test])
np.savetxt('discriminant-unscaled.csv', res.T, delimiter = ',')

# Discriminant Distribution
plots.distribution(y_pred, y_test, xlabel = 'Discriminant', bins = 50)
plt.yscale('log')
atlasify('work in progress')
plt.show()