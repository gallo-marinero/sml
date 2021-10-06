import sys, os, matplotlib, time, os.path
import numpy as np
#import joblib
import pandas as pd
#import statistics as st
#from matplotlib import use
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn import svm 
from sklearn.model_selection import cross_val_score
import pca, train_mod
#from sklearn.metrics import pairwise_distances, davies_bouldin_score
#from umap import UMAP

# Create StandardScaler instance
sscaler=StandardScaler()

# Insert the y (output or target column title)
#yname='Dry_thickness'
yname='Weight_AM'
# Title of the columns that are to be dropped (if none, leave empty dropcols=[])
dropcols=['Airflow2','Jar','Sample','Exp','Active_Material','Liquid_content','Carbon','Binder','Dry_thickness']
#dropcols=['Jar','Sample','Exp','Time1','Time2','Time3','Width','T1','T2','Airflow2','Active_Material','Liquid_content','Carbon','Binder','Dry_thickness']
# Read the raw numbers as pd.DataFrame
rawdat=pd.read_csv(sys.argv[1])
# Loop over the columns that should be dropped
for i in dropcols:
    rawdat=rawdat.drop(columns=[i])
# Drop the column that is the output and assign it to rawy
y=rawdat[yname]
x=rawdat.drop(columns=[yname])
# Store the name of the columns in a list (useful for plotting e.g.: feature_importances_)
feat_names=list(x.columns)
# Perform standard scaling before PCA
x_scal=sscaler.fit_transform(x)
# Perform PCA análisis and choose the most relevant components
# if crit (second argument) is integer -> perform PCA with crit number of components
# if crit is float -> perform PCA until crit variance is explained
# components
pca.pca_feat(x_scal,.95,True)
# Train models and get scores. x,y are not scaled
train_mod.trainmod(x,y,feat_names)
# Build a pipeline for simplicity and safety when doing the steps to build the
# models
#pipe = make_pipeline(StandardScaler(),svm.SVR())
#pipe = make_pipeline(StandardScaler(),SGDR(max_iter=1500))
#print(pipe.fit(x_train,y_train))
#print(pipe.score(x_train,y_train))
