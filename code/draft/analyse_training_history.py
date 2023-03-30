"""
@author: liza mijovic

Script to analyse DeepLearn2023 ANN results:
- loss function

"""


import matplotlib.pyplot as plt
import pickle


# crank up font size 
plt.rcParams.update({'font.size': 16})

#-----------------------------------------------------------------------
# 1,2 
# loss plot for classifier only: 
with open('clf_standalone_history.pckl', "rb") as file_pi:
    history = pickle.load(file_pi)

clf_loss_train = history['loss']
clf_loss_val = history['val_loss']
plt.plot(clf_loss_train, 'orange', label = 'Training loss')
plt.plot(clf_loss_val, 'blue', label = 'Validation loss')
plt.xlabel('Epochs',horizontalalignment='right', x=1.0)
plt.ylabel('Classifier Loss',horizontalalignment='right', y=1.0)
plt.legend(frameon = False)
plt.savefig("clf_standalone_loss.png", bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------
# 3 ANN:
# Loss plotsfor ANN
with open('ANN_history.pckl', "rb") as file_pi:
    history = pickle.load(file_pi)

# Combined model loss 
loss_train = history['loss']
loss_val = history['val_loss']
plt.plot(loss_train, 'orange', label='Training loss')
plt.plot(loss_val, 'blue', label='Validation loss')
plt.xlabel('Epochs',horizontalalignment='right', x=1.0)
plt.ylabel('ANN Loss',horizontalalignment='right', y=1.0)
plt.legend(frameon = False)
plt.savefig("loss_ANN.png", bbox_inches='tight')
plt.show()

# Classifier loss plot
clf_loss_train = history['classifier_loss']
clf_loss_val = history['val_classifier_loss']
plt.plot(clf_loss_train, 'orange', label='Training loss')
plt.plot(clf_loss_val, 'blue', label='Validation loss')
plt.xlabel('Epochs',horizontalalignment='right', x=1.0)
plt.ylabel('ANN Class. Loss',horizontalalignment='right', y=1.0)
plt.legend(frameon = False)
plt.savefig("loss_ANN_classifier.png", bbox_inches='tight')
plt.show()

# Adversary loss plot
adv_loss_train = history['classifier_loss']
adv_loss_val = history['val_classifier_loss']
plt.plot(adv_loss_train, 'orange', label='Training loss')
plt.plot(adv_loss_val, 'blue', label='Validation loss')
plt.xlabel('Epochs',horizontalalignment='right', x=1.0)
plt.ylabel('ANN Adversary Loss',horizontalalignment='right', y=1.0)
plt.legend(frameon = False)
plt.savefig("loss_ANN_adversary.png", bbox_inches='tight')
plt.show()

exit(0)
