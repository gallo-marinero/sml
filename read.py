import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer
from sklearn.model_selection import KFold
from importlib import import_module
import sys,plot, train, tests, fs, al_boss, os,pp
#from umap import UMAP

# Define input variables
vrbls=['classification','al','estimator','var_thresh','feat_sel',\
        'bins','plot_dat_hist','umap','simil_test','spline_knots',\
        'feat_names','short_score','yname','dropcols','data_f','gen_feat',\
        'pca_expl_var']

# ~ Defaults definition~
# Regression problem by default
classification=False
al=False
# Available estimators:
# Linear Regression (Tweedie Regressor): linr
# K-Nearest Neighbors: knn
# Support Vector Machine: svm
# Stochastic Gradient Descent: sgd
# Decission Tree: dt
# Logistic Regression (classification only): lr
# Gaussian process: gp # NOT READY!
# Neural network: nn
# Default estimator:
estimator=['dt']
# Some preprocessing
# gen_feat contains the degree
gen_feat=False
# Whether to use spline transformation (polynomial is default)
# Use value as knots
spline_knots=False
# Remove features repeated >80% of the time
var_thresh=False
# Perform feature selection
feat_sel=False
# Plot data in histograms
plot_dat_hist=False
bins=10
plot_dat_scatter=False
#ref='T_annealing'
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
# Cross-validation settings
cv=KFold(n_splits=10,shuffle=True,random_state=42)

# Read input from file
# Add path where code is executed to be able to load the input file as a module
sys.path.append(os.getcwd())
# Import variables in input file
inp_f=__import__(sys.argv[1])
# For all variables, if present in the input file, overwrite default value
for i in vrbls:
# Check if the variable is present in input file
    if hasattr(inp_f,i):
# Update variable
        globals()[i] = getattr(inp_f,i)

if al:
    print('\n     ~ Active learning task ~')
    print('\n     ------------------------\n')
    al_boss.boss(data_f)
    exit()
else:
    if classification:
        print('\n    ~ Classification fit ~')
        print('\n    ----------------------')
    else:
        print('\n    ~ Regression fit ~')
        print('\n    ------------------')
print('\n -> Reading the data from', data_f)
print('\n -> Modelling \'', yname, '\' variable')
print('\n -> Droping features: ')
for i in dropcols:
    print('    ',i)
print('\n -> Using estimator: ')
for i in estimator:
    print('    ',i)

# Create StandardScaler instance
#scal=StandardScaler()
#scal=MinMaxScaler()
scal=PowerTransformer()

x=pd.read_csv(data_f,sep=',')
# Loop over the columns that should be dropped
for i in dropcols:
    x=x.drop(columns=[i])
# Pop the column that is the output and assign it to y
y=x.pop(yname)
# Store the name of the columns in a list (useful for plotting e.g.: feature_importances_)
feat_names=list(x.columns)
n_feats=len(x.columns)

# Plot data in histogram form
if plot_dat_hist:
    plot.plot_hist(x,bins,feat_names,scal=False)
# If transformation, apply it and return back the labels
    x_scal=scal.fit_transform(x)
    x=pd.DataFrame(x_scal,index=x.index,columns=x.columns)
    plot.plot_hist(x,bins,feat_names,scal=scal.__class__.__name__)
    if not plot_dat_scatter:
        exit()
if plot_dat_scatter:
    plot.plot_scatter(x,ref,feat_names)
    exit()

# Call preprocesing
if gen_feat:
    if spline_knots:
        x=pp.spline_features(x,gen_feat,spline_knots)
    else:
        x=pp.polynomial_features(x,gen_feat)

# Load and call active learning module
if al:
    import al
    al.train(x,y,estimator[0],classification)
    exit()

# Remove features that are 0/1 in 80% of the cases
if var_thresh:
    dropped=fs.vt(x,feat_names)
# Drop them from the original dataset. Just to be able to recover the original tags
    for i in dropped:
        x=x.drop(columns=[i])
# Perform scaling
if scal:
    print('\n -> Using scaling:')
    print('     ',scal.__class__.__name__)
    x_scal=scal.fit_transform(x)
# Tag back the columns after scaling
    x=pd.DataFrame(x_scal,index=x.index,columns=x.columns)

# Perform PCA analysis and choose the most relevant components
# if crit (second argument) is integer -> perform PCA with crit number of components
# if crit is float -> perform PCA until crit variance is explained
# components
if pca_expl_var != 0:
    x=plot.pca_feat(x,pca_expl_var,True)
# Perform UMAP dim reduction and plot it (coloured with the results)
if umap:
    plot.umap(x,y)
# Perform train and test similarity tests
if simil_test: 
    print('\n-> Performing similarty tests')
    tests.rf_diff_dist(x,y,yname)
    tests.ks(x,y)
##tests.mahalanobis(x,y)

# Perform feature selection
if feat_sel:
    print('\n-> Performing feature selection')
# Perform MI and F-reg. Select according to SelectKBest
    fs.skb(x,y,feat_names,short_score,classification,estimator,cv)
# Perform recursive feature elimination with cross-validation
    fs.rfecv(x,y,feat_names,short_score,classification,estimator,cv)

if n_feats > x.shape[1]:
    print('\n  Applied feature reduction')
    print('   Reduced from ',n_feats,' to ',x.shape[1],' features')

print('\n -> Modelling with the following',x.shape[1],'features:')
for i in feat_names:
    print('     ', i)
# If it is no active learning task, train models and get scores. x,y are not scaled
train.trainmod(x,y,feat_names,short_score,classification,estimator,cv)

# Evaluate model
##predict_evaluate.pred_eval(x,y,model)
# Build a pipeline for simplicity and safety when doing the steps to build the
# models
#pipe = make_pipeline(StandardScaler(),svm.SVR())
#pipe = make_pipeline(StandardScaler(),SGDR(max_iter=1500))
#print(pipe.fit(x_train,y_train))
#print(pipe.score(x_train,y_train))
