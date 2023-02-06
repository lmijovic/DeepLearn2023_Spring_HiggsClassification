import pandas as pd
import pylorentz as pl

def get_mass(pt_y1, eta_y1, phi_y1, e_y1,
             pt_y2, eta_y2, phi_y2, e_y2):
    mom_y1 = pl.Momentum4.e_eta_phi_pt(e_y1, eta_y1, phi_y1, pt_y1)
    mom_y2 = pl.Momentum4.e_eta_phi_pt(e_y2, eta_y2, phi_y2, pt_y2)
    return (mom_y1+mom_y2).m

my_cols=['pt_y1','eta_y1','phi_y1','e_y1',
         'pt_y2','eta_y2','phi_y2','e_y2',
         'pt_j1','eta_j1','phi_j1','e_j1','dlr1_j1',
         'pt_j2','eta_j2','phi_j2','e_j2','dlr1_j2',
         'pt_j3','eta_j3','phi_j3','e_j3','dlr1_j3',
         'pt_j4','eta_j4','phi_j4','e_j4','dlr1_j4',
         'pt_j5','eta_j5','phi_j5','e_j5','dlr1_j5',
         'pt_j6','eta_j6','phi_j6','e_j6','dlr1_j6',
         'pt_j7','eta_j7','phi_j7','e_j7','dlr1_j7',
         'pt_j8','eta_j8','phi_j8','e_j8','dlr1_j8',
         'pt_j9','eta_j9','phi_j9','e_j9','dlr1_j9',
         'pt_j10','eta_j10','phi_j10','e_j10','dlr1_j10',
         'pt_j20','eta_j20','phi_j20','e_j20','dlr1_j20',
         'pt_j30','eta_j30','phi_j30','e_j30','dlr1_j30',
         'pt_j40','eta_j40','phi_j40','e_j40','dlr1_j40',
         'pt_j50','eta_j50','phi_j50','e_j50','dlr1_j50',
         'pt_j60','eta_j60','phi_j60','e_j60','dlr1_j60',
         'pt_j70','eta_j70','phi_j70','e_j70','dlr1_j70',
         'pt_j80','eta_j80','phi_j80','e_j80','dlr1_j80']

sig=pd.read_csv('sig.csv',names=my_cols)
bkg=pd.read_csv('bkg.csv',names=my_cols)

#ensure sig/bkg label is 1/0
sig['label']=1
bkg['label']=0

# concatenate sig & background, randomize
data = pd.concat([sig,bkg], ignore_index=True)

todrop = ['dlr1_j1',
          'dlr1_j2',
          'dlr1_j3',
          'dlr1_j4',
          'pt_j5','eta_j5','phi_j5','e_j5','dlr1_j5',
          'pt_j6','eta_j6','phi_j6','e_j6','dlr1_j6',
          'pt_j7','eta_j7','phi_j7','e_j7','dlr1_j7',
          'pt_j8','eta_j8','phi_j8','e_j8','dlr1_j8',
          'pt_j9','eta_j9','phi_j9','e_j9','dlr1_j9',
          'pt_j10','eta_j10','phi_j10','e_j10','dlr1_j10',
          'pt_j20','eta_j20','phi_j20','e_j20','dlr1_j20',
          'pt_j30','eta_j30','phi_j30','e_j30','dlr1_j30',
          'pt_j40','eta_j40','phi_j40','e_j40','dlr1_j40',
          'pt_j50','eta_j50','phi_j50','e_j50','dlr1_j50',
          'pt_j60','eta_j60','phi_j60','e_j60','dlr1_j60',
          'pt_j70','eta_j70','phi_j70','e_j70','dlr1_j70',
          'pt_j80','eta_j80','phi_j80','e_j80','dlr1_j80']
for vdrop in todrop:
    data = data.drop([vdrop], axis=1)

data['myy'] = data.apply(lambda row : get_mass(row['pt_y1'],row['eta_y1'],row['phi_y1'],row['e_y1'],
                                               row['pt_y2'],row['eta_y2'],row['phi_y2'],row['e_y2']), axis = 1)
print(data.head(20))
# randomize rows such that dataframe has uniform properties
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv('data.csv')
