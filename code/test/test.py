import layers

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd 

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


np.random.seed(42)

#=======================================================================
# main
#=======================================================================

#-----------------------------------------------------------------------
# Import data
data=pd.read_csv('data_test.csv')
print('Imported data:')
print(data.head(4))

# put columns into numpy nd-arrays 
# myy is the fit variable
myy = data['myy'].values
# signal/background label is 1/0
y = data['label'].values
# classifier inputs are photon 4-momenta
X = data[['pt_y1','eta_y1','phi_y1','e_y1',
         'pt_y2','eta_y2','phi_y2','e_y2']].values

# Normalize X data
normalize = StandardScaler()
X = normalize.fit_transform(X)
      
# Split data into training and testing sets
X_train, X_test, y_train, y_test, myy_train, myy_test = train_test_split(X, y, myy, test_size = 0.30, random_state = 42)

#------------------------------------------------------------------------
print('\n =============== 1) Standalone classifier  =============== \n ')
# Number of samples, features, epochs & batch size
num_samples = X_train.shape[0]
num_feat = X_train.shape[1]
num_epochs = 20
batch = 500

clf_standalone = classifier(num_feat)
clf_standalone.compile(optimizer='adam', loss='binary_crossentropy')
hist_clf_standalone = clf_standalone.fit(X_train, y_train,
                                         epochs = num_epochs, batch_size = batch,
                                         validation_split = 0.2, verbose = 2)

# generate predictions & store results 
y_pred = clf_standalone.predict(X_test).flatten()
res = np.array([myy_test.T, y_pred, y_test])
np.savetxt('clf_results.csv', res.T, delimiter = ',')

#------------------------------------------------------------------------
print('\n ================== 3) Adversarial training ================ \n ')


# Define parameters for combined network
lambda_reg = 3             # Regularization parameter 
num_gmm = 5                # Number of GMM components
lr = 1e-5                 # Relative learning rates for classifier and adversary

loss_weights = [lr, lambda_reg]

# Prepare sample weights (i.e. only do mass-decorrelation for background)
sample_weight = [np.ones(num_samples, dtype=float), (y_train == 0).astype(float)]
sample_weight[1] *= np.sum(sample_weight[0])/ np.sum(sample_weight[1])   

# Rescale diphoton invariant mass to [0,1]
sc_myy_train = myy_train - myy_train.min()
sc_myy_train /= myy_train.max()

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
Generate output files
'''

# Test set predictions
y_pred = clf.predict(X_test).flatten()

# Write myy, predictions and label to file
res = np.array([myy_test.T, y_pred, y_test])
np.savetxt('ANN_results.csv', res.T, delimiter = ',')

#------------------------------------------------------------------------
print('\n =============== Run Finished  =============== \n ')
print('\n Check if the following files are non-empty: \n'   )
print('  - \n clf_results.csv'   )
print('  - \n ANN_results.csv'   )
