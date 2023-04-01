import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb

'''
tested with python3

'''

plt.rcParams.update({'font.size': 20})


#----------------------------------------------------------------------------- 
fname="data_scaled_2M.csv" # input file 

#-----------------------------------------------------------------------------
# supress index column:
df=pd.read_csv(fname, index_col=0)

print('--------------------- working with file -----------------------')
print(df.head(5))


# bkg events only
df=df.loc[df['label']==0]

df=df.drop(["label"],axis=1)
# drop 2nd photon for visibility:
df=df.drop(["pt_y2"],axis=1)
df=df.drop(["eta_y2"],axis=1)
df=df.drop(["phi_y2"],axis=1)
df=df.drop(["e_y2"],axis=1)

plt.figure()
f = plt.figure(figsize=(20, 15))
corr = df.corr()
#sb.set_palette("vlag")
#sb.color_palette("coolwarm", as_cmap=True)
sb.heatmap(corr, cmap="coolwarm",
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,
           annot=True)
#plt.text(0,0, "DL2023",
#         rotation='horizontal', verticalalignment = 'bottom' , horizontalalignment = 'left', fontsize=30 )
plt.savefig("corr.png", bbox_inches='tight')
plt.savefig("corr.pdf", bbox_inches='tight')
plt.show()

exit(0)
