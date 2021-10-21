import sys, os, matplotlib, time, os.path
import numpy as np
#import joblib
import pandas as pd
#import statistics as st
#from matplotlib import use
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn import svm 
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest as skb
import dim_red_plot, train_mod, tests, fs, predict_evaluate
#from sklearn.metrics import pairwise_distances, davies_bouldin_score
#from umap import UMAP

# Create StandardScaler instance
#sscaler=StandardScaler()
sscaler=MinMaxScaler()

# Insert the y (output or target column title)
#yname='Dry_thickness'
yname='OCV_after_24h_rest'
# Title of the columns that are to be dropped (if none, leave empty dropcols=[])
#dropcols=['Airflow2','Jar','Sample','Exp','Active_Material','Liquid_content','Carbon','Binder','Dry_thickness']
dropcols=['Pouch_cell','Anode_thickness','DCH_capacity3_cycle_mass','DCH_capacity3_cycle_area',
        'Coulombic_efficiency3_cycle','AverageCH_voltage3_cycle', 'AverageDCH_voltage3_cycle',
        'Polarization_20%SOC_3cycle','Polarization_80%SOC_3cycle','DCH_capacity_last_mass',
        'DCH_capacity_last_area','Coulombic_efficiency_last',
        'Average_CH_voltage_last','Average_DCH_voltage_last','Polarization_20%SOC_last',
        'Polarization_80%SOC_last','Number_cycles','Number_cyclesQE>99%','Number_cyclesQretention>80%']
# Read the raw numbers as pd.DataFrame
rawdat=pd.read_csv(sys.argv[1])
# Loop over the columns that should be dropped
for i in dropcols:
    rawdat=rawdat.drop(columns=[i])
# Pop the column that is the output and assign it to y
rawdata=rawdat
y=rawdat.pop(yname)
x=rawdat
# Store the name of the columns in a list (useful for plotting e.g.: feature_importances_)
feat_names=list(x.columns)
# Perform standard scaling before PCA
x_scal=sscaler.fit_transform(x)
x_scal=pd.DataFrame(x_scal,index=x.index,columns=x.columns)
# Perform PCA analysis and choose the most relevant components
# if crit (second argument) is integer -> perform PCA with crit number of components
# if crit is float -> perform PCA until crit variance is explained
# components
'''
dim_red_plot.pca_feat(x_scal,.95,True)
# Perform UMAP dim reduction and plot it (coloured with the results)
dim_red_plot.umap(x_scal,y)
exit()
'''
'''
# Perform train and test similarity tests
print('\n-> Performing similarty tests')
tests.rf_diff_dist(x_scal,y,yname)
tests.ks(x_scal,y)
#tests.mahalanobis(x,y)
exit()
'''
# Perform feature selection
print('\n-> Performing feature selection')
fs.skb(x,y)
exit()
# Train models and get scores. x,y are not scaled
train_mod.trainmod(x,y,feat_names=False)
# Evaluate model
predict_evaluate.lr(x,y)
# Build a pipeline for simplicity and safety when doing the steps to build the
# models
#pipe = make_pipeline(StandardScaler(),svm.SVR())
#pipe = make_pipeline(StandardScaler(),SGDR(max_iter=1500))
#print(pipe.fit(x_train,y_train))
#print(pipe.score(x_train,y_train))
