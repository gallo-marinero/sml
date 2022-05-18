import pandas as pd
import numpy as np
from random import seed
from random import choice

dat=[]
seed(1)
size=40
keys=['solid_content']
sequence=[[25,30,32,34,36,38,40,48,50,52]]
keys.append('wet_thickness')
sequence.append([1200,1300,1400,1600,1700,1800,1900,2100,\
        2200,2300,2400,2500,2600,2700,2800,2900,3000])
keys.append('web_speed')
sequence.append(np.arange(.3,.8,.3))
keys.append('viscosity')
sequence.append(np.arange(.5,10.05,.1))
keys.append('flow_z1')
sequence.append(np.arange(2,6.5,2))
keys.append('flow_z2')
sequence.append([2,4,8])
keys.append('drying_z1')
sequence.append([27,29,31,33,37,39])
keys.append('drying_z2')
sequence.append([30,45])
keys.append('drying_speed')
sequence.append([.1,.3,.6,.7,.8,.9,1])
keys.append('shear_rate')
sequence.append([0])

sample=pd.DataFrame()
for j in range(size):
    for i in sequence:
        dat.append(np.round(choice(i),3))
    sample=pd.concat([sample,pd.DataFrame(dat).T])
    dat=[]
sample.columns=keys
sample.to_csv('sample_collection.csv',index=False)
