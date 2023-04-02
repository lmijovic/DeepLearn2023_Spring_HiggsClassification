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
data=pd.read_csv('../clf_standalone_results.csv',names=cols)
sig=data[data['y']==1]
bkg=data[data['y']==0]


#------------------------------------------------------------------------
# ROC curve:
# generated from actual ('y') vs predicted ('y_pred') labels 
fpr, tpr, thresholds = roc_curve(data['y'], data['y_pred'], pos_label=1)
roc_auc = auc(fpr,tpr)

plt.figure()
lw=1
plt.plot(fpr, tpr, color='orange',
         lw=lw, label='Area U.C. = %0.2f' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate',horizontalalignment='right', x=1.0)
plt.ylabel('True Positive Rate',horizontalalignment='right', y=1.0)
plt.savefig("roc.png", bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------


#------------------------------------------------------------------------
# discriminant distribution for S and B
plt.figure()
bins = np.linspace(0,1,51)

plt.hist(sig['y_pred'], bins, color='red',
         histtype='step',
         label=r'H$\rightarrow \gamma\gamma$ signal',
         weights=get_w(sig['y_pred']))

plt.hist(bkg['y_pred'], bins, color='blue',
         histtype='step',
         label='background',
         weights=get_w(bkg['y_pred']))

plt.xlabel('ML discriminant',horizontalalignment='right', x=1.0)
plt.ylabel('Fraction of events/0.02',horizontalalignment='right', y=1.0)
plt.legend(frameon = False)
plt.savefig("discriminant.png", bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------


#------------------------------------------------------------------------
# background
plt.figure()
bins = np.linspace(105,170,65)

high_disc_bkg=bkg[bkg['y_pred']>=0.6]

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
plt.show()
#------------------------------------------------------------------------


exit(0)
