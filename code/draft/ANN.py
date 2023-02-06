import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
import tensorflow
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model

import layers

print("keras version")
print(keras.__version__)
print("tensorflow version")
print(tensorflow.__version__)


def classifier(num_feat):
    # Inputs
    i = Input(shape = (num_feat,))
    
    # Hidden layers
    x1 = Dense(24, activation = 'relu')(i)      
    x2 = Dense(16, activation = 'relu')(x1)     
    x3 = Dense(8, activation = 'relu')(x2)      
    
    # Output layer
    o = Dense(1, activation = 'sigmoid')(x3)
    
    # Build NN classifier
    return Model(inputs = i, outputs = o, name = 'classifier')


def adversary(num_gmm):
    # Inputs
    i = Input(shape = (1,))
    myy = Input(shape = (1,))
    
    # Hidden layers
    x1 = Dense(200, activation = 'relu')(i)       
    x2 = Dense(100, activation = 'relu')(x1)      
    x3 = Dense(50, activation = 'relu')(x2)      
    
    # Gaussian mixture model (GMM) components
    coeffs = Dense(num_gmm, activation='softmax')(x3)  # GMM coefficients sum to one
    means  = Dense(num_gmm, activation='sigmoid')(x3)  # Means are on [0, 1]
    widths = Dense(num_gmm, activation='softplus')(x3)  # Widths are positive
    
    # Posterior probability distribution function
    pdf = layers.PosteriorLayer(num_gmm)([coeffs, means, widths, myy])

    return Model(inputs = [i, myy], outputs = pdf, name = 'adversary')


def combined(clf, adv, lambda_reg, lr_ratio):
    # Inputs
    clf_input = Input(shape = clf.layers[0].input_shape[0][1])
    myy_input = Input(shape = (1,))
    
    # Classifier ouput
    clf_output = clf(clf_input)
    
    # Gradient reversal
    gradient_reversal = layers.GradientReversalLayer(lambda_reg * lr_ratio)(clf_output)
    
    # Adversary
    adv_output = adv([gradient_reversal, myy_input])
    
    return Model(inputs=[clf_input, myy_input], outputs=[clf_output, adv_output], name='combined')


def custom_loss(y_true, y_pred):
    '''
    Kullback-Leibler loss; maximises posterior p.d.f.
    Equivalent to binary-cross-entropy for all y = 1
    '''    
    return -K.log(y_pred)


np.random.seed(2)


'''
Data pre-processing
'''
# Import data
#data = np.loadtxt('x.csv', delimiter = ',', skiprows=1)

# Import data
#data = np.loadtxt('data_nohead.csv', delimiter = ',',skiprows=1)
data = np.genfromtxt('data.csv', delimiter = ',', skip_header=1, filling_values=0)

myy = data[:,26]
y = data[:,25]
X = data[:,1:25]

# Normalize X data
normalize = StandardScaler()
X = normalize.fit_transform(X)
      
# Split data into training and testing sets
X_train, X_test, y_train, y_test, myy_train, myy_test = train_test_split(X, y, myy, test_size = 0.20, random_state = 5)

# Rescale diphoton invariant mass to [0,1]
sc_myy_train = myy_train - myy_train.min()
sc_myy_train /= myy_train.max()


'''
Define parameters for combined network
'''

# Number of samples, features, epochs & batch size
num_samples = X_train.shape[0]
num_feat = X_train.shape[1]
num_epochs = 100
batch = 5000

lambda_reg = 3             # Regularization parameter 
num_gmm = 5                # Number of GMM components
lr = 1e-5                 # Relative learning rates for classifier and adversary

loss_weights = [lr, lambda_reg]

# Prepare sample weights (i.e. only do mass-decorrelation for background)
sample_weight = [np.ones(num_samples, dtype=float), (y_train == 0).astype(float)]
sample_weight[1] *= np.sum(sample_weight[0])/ np.sum(sample_weight[1])   


'''
Define classifier, adversary & combined network
'''

clf = classifier(num_feat)
adv = adversary(num_gmm)
ANN = combined(clf, adv, lambda_reg, lr)

# Build & train combined model
ANN.compile(optimizer='adam', loss=['binary_crossentropy', custom_loss], loss_weights = loss_weights)
hist_ANN = ANN.fit([X_train, sc_myy_train], [y_train, np.ones_like(sc_myy_train)], 
                   sample_weight = sample_weight, epochs = num_epochs, batch_size = batch, 
                   validation_split = 0.2, verbose = 2)


'''
Generate plots & output files
'''

# Test set predictions
y_pred = clf.predict(X_test).flatten()

# Write myy, predictions and label to file
res = np.array([myy_test.T, y_pred, y_test])
np.savetxt('discriminant.csv', res.T, delimiter = ',')

# Loss plot 
loss_train = hist_ANN.history['loss']
loss_val = hist_ANN.history['val_loss']
epochs = range(1,len(loss_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(frameon = False)
plt.show()

# Classifier loss plot
clf_loss_train = hist_ANN.history['classifier_loss']
clf_loss_val = hist_ANN.history['val_classifier_loss']
plt.plot(epochs, clf_loss_train, 'g', label = 'Classifier training loss')
plt.plot(epochs, clf_loss_val, 'b', label = 'Classifier validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(frameon = False)
plt.show()

# Adversary loss plot
adv_loss_train = hist_ANN.history['adversary_loss']
adv_loss_val = hist_ANN.history['val_adversary_loss']
plt.plot(epochs, adv_loss_train, 'g', label = 'Adversary training loss')
plt.plot(epochs, adv_loss_val, 'b', label = 'Adversary validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(frameon = False)
plt.show()
