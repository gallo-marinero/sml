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

# Remove features repeated >80% of the time
var_thresh=False
# Perform feature selection
feat_sel=True
# Print UMAP dimensionality reduction
umap=False
# Performed PCA explained variance. If == 0, skip
pca_expl_var=0
# Performed similarity tests
simil_test=False
# Short or long scores
short_score=True
# Pass feature names
feat_names=False

# Create StandardScaler instance
#sscaler=StandardScaler()
sscaler=MinMaxScaler()

#POUCH
#yname='cathode_am_mass'
#yname='n_cycles_qretention80'
# Title of the columns that are to be dropped (if none, leave empty dropcols=[])
#dropcols=['Airflow2','Jar','Sample','Exp','Active_Material','Liquid_content','Carbon','Binder','Dry_thickness']
#pouchdropcols=['pouch_cell','n_cycles_qe99']
#pouchdropcols=['pouch_cell','anode_thickness','membrane_fabrication','membrane_composition','manual_sealing','viscosity','web_speed','drying_z1','flow_z1','drying_z2','flow_z2','wet_thickness','drying_speed','alu_mass','n_cycles_qe99','rate_capability']
#dropcols=['pouch_cell']
# AJURIA
#yname='weightam'
#dropcols=['Exp','dry_thickness']
# am_loading
yname='am_loading'
dropcols=['name','solid_content']#,'visual_inspection']
# Read the raw numbers as pd.DataFrame
print('-> Reading the data from', sys.argv[1])
print('   Modelling', yname)
x=pd.read_csv(sys.argv[1])
# Loop over the columns that should be dropped
for i in dropcols:
    x=x.drop(columns=[i])
# Pop the column that is the output and assign it to y
y=x.pop(yname)
# Store the name of the columns in a list (useful for plotting e.g.: feature_importances_)
feat_names=list(x.columns)
n_feats=len(x.columns)

# Remove features that are 0/1 in 80% of the cases
if var_thresh:
    x,dropped=fs.vt(x,feat_names)
# Drop them from the original dataset. Just to be able to recover the original tags
    for i in dropped:
        x=x.drop(columns=[i])
# Perform standard scaling
x_scal=sscaler.fit_transform(x)
# Tag back the columns after scaling
x_scal=pd.DataFrame(x_scal,index=x.index,columns=x.columns)

# Perform PCA analysis and choose the most relevant components
# if crit (second argument) is integer -> perform PCA with crit number of components
# if crit is float -> perform PCA until crit variance is explained
# components
if pca_expl_var != 0:
    x=dim_red_plot.pca_feat(x_scal,pca_expl_var,True)
# Perform UMAP dim reduction and plot it (coloured with the results)
if umap:
    dim_red_plot.umap(x_scal,y)
# Perform train and test similarity tests
if simil_test: 
    print('\n-> Performing similarty tests')
    tests.rf_diff_dist(x_scal,y,yname)
    tests.ks(x_scal,y)
##tests.mahalanobis(x,y)

# Perform feature selection
if feat_sel:
    print('\n-> Performing feature selection')
# Perform MI and F-reg. Select according to SelectKBest
    fs.skb(x,y,feat_names,short_score)
# Perform recursive feature elimination with cross-validation
    fs.rfecv(x,y,feat_names,short_score)

if n_feats > x.shape[1]:
    print('\n  Applied feature reduction PCA')
    print('   Reduced from ',n_feats,' to ',x.shape[1],' features')

print('\n -> Modellling with ',x.shape[1],' features')
# Train models and get scores. x,y are not scaled
train_mod.trainmod(x,y,feat_names,short_score)

# Evaluate model
##predict_evaluate.pred_eval(x,y,model)
# Build a pipeline for simplicity and safety when doing the steps to build the
# models
#pipe = make_pipeline(StandardScaler(),svm.SVR())
#pipe = make_pipeline(StandardScaler(),SGDR(max_iter=1500))
#print(pipe.fit(x_train,y_train))
#print(pipe.score(x_train,y_train))
