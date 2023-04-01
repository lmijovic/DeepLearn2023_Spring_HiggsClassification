"""
Analyse DeepLearn2023 ANN results:
- ROC curve
- discriminant distributions
- mass sculpting

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# crank up font size 
plt.rcParams.update({'font.size': 16})


# get weights to normalize an array to unit area:
def get_w(p_arr):
    sum_weights=float(len(p_arr))
    numerators = np.ones_like(p_arr)
    return(numerators/sum_weights)



#=======================================================================
# main
#=======================================================================

# import data; written as:
## Write myy, predictions and label to file
#res = np.array([myy_test.T, y_pred, y_test])
#np.savetxt('discriminant.csv', res.T, delimiter = ',')

cols=['myy','y_pred','y']
data=pd.read_csv('clf_standalone_results.csv',names=cols)
sig=data[data['y']==1]
bkg=data[data['y']==0]


#------------------------------------------------------------------------
# background
plt.figure()
bins = np.linspace(105,170,65)

high_disc_bkg=bkg[bkg['y_pred']>=0.6]

plt.hist(sig['myy'], bins, color='red',
         histtype='step',
         label=r'signal',
         weights=get_w(sig['myy']))

plt.hist(high_disc_bkg['myy'], bins, color='orange',
         histtype='step',linestyle='--',
         label=r'background, D>0.6',
         weights=get_w(high_disc_bkg['myy']))

plt.hist(bkg['myy'], bins, color='blue',
         histtype='step',
         label='all background',
         weights=get_w(bkg['myy']))

plt.xlabel(r'm$_{\gamma\gamma}$ [GeV]',horizontalalignment='right', x=1.0)
plt.ylabel('Fraction of events/1 GeV',horizontalalignment='right', y=1.0)
plt.legend(frameon = False)
plt.savefig("myy.png", bbox_inches='tight')
plt.savefig("myy.pdf", bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------


exit(0)
