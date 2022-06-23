import pandas as pd
import numpy as np
from random import seed
from random import choice
# Selection without replacement:
#once an item from the list is selected for the subset, it is not added back to
#the original list (i.e. is not made available for re-selection)
from random import sample 

dat=[]
seed(1)
size=30
'''
# Maica
keys=['solid_content'] # (20,52)
sequence=[[25,30,32,34,36,38,40,48,50,52]]
keys.append('wet_thickness') # (50,600) (50,1500) if drying_speed=0
sequence.append([1200,1300,1400,1600,1700,1800,1900,2100,\
        2200,2300,2400,2500,2600,2700,2800,2900,3000])
keys.append('web_speed')
sequence.append(np.arange(.1,1,.3))
keys.append('viscosity')
sequence.append(np.arange(.5,10.05,.25))
keys.append('flow_z1')
sequence.append(np.arange(2,6.5,2))
keys.append('flow_z2')
sequence.append([2,4,8])
keys.append('drying_z1')
sequence.append([25,30,35,40])
keys.append('drying_z2')
sequence.append([25,30,45])
keys.append('drying_speed')
sequence.append([0,1])
keys.append('shear_rate')
sequence.append([0])
'''

keys=['li_ratio']
sequence=[np.arange(0,.5,.05)]
keys.append('time')
sequence.append(np.arange(5,10,1))
keys.append('t_gel')
sequence.append(np.arange(2,5,.2))
keys.append('t_agitacion')
sequence.append(np.arange(1,3,.2))
keys.append('T_rampa_enf')
sequence.append(np.arange(2,7,1))
keys.append('T_rampa_cal')
sequence.append(np.arange(5,10,1))
keys.append('stech_li_excess')
sequence.append(np.arange(.02,.04,.01))
keys.append('T_annealing')
#sequence.append(np.arange(800,950,50))
t=list(np.arange(800,1000,50))
keys.append('position_muffle')
#sequence.append(muffle)

new_sample=pd.DataFrame()
for j in range(len(t)):
    muffle=np.arange(1,13,1)
    t_sample=choice(t)
    t.remove(t_sample)
    for k in muffle:
        for i in sequence:
            dat.append(np.round(choice(i),3))
        dat.append(t_sample)
        dat.append(k)
        new_sample=pd.concat([new_sample,pd.DataFrame(dat).T])
        dat=[]
new_sample.columns=keys
new_sample.to_csv('sample_collection.csv',index=False)
