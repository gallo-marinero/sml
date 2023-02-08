import pandas as pd
from os.path import exists
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# Perform PCA component analysis
def pca_feat(x,crit,v):
# If an integer is given, perform PCA for crit number of components
    if isinstance(crit,int):
        print('\n-> Performing PCA analysis with',crit,'components')
# If a float is given, perform PCA until crit variance ratio is achieved
    elif isinstance(crit,float):
        print('\n-> Performing PCA analysis until',crit*100,'% of variance is explained')
    pca=PCA(crit)
    x_pca=pca.fit_transform(x)
    if v:
        print('\n  The',pca.n_components_,'components explain',
                round(100*sum(pca.explained_variance_ratio_),2),'% of the variance')
        for i in range(pca.n_components_):
            print('  ',i+1,round(100*pca.explained_variance_ratio_[i],2))

    plt.bar(np.arange(pca.n_components_)+1,pca.explained_variance_ratio_*100,alpha=.5)
    plt.xlabel('Component')
    plt.ylabel('% of explained component')
    plt.text(pca.n_components_-.7,round(pca.explained_variance_ratio_[0]*90,1),'Total '
            +str(round(sum(100*pca.explained_variance_ratio_),2))+'%')
    for i in range(pca.n_components_):
        plt.annotate(round(100*pca.explained_variance_ratio_[i],2),
                xy=(i+1,pca.explained_variance_ratio_[i]*100), ha='center', va='bottom')

    plt.savefig('explainedvar_pca.png')
    plt.clf()
    return x_pca

def umap(x,y):
    print('\n-> Applying UMAP dimensionality reduction')
#    plt.title(title)
    umap=UMAP(random_state=42)
    umap_data=umap.fit_transform(x)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(umap_data[:, 0], umap_data[:, 1], s=9,
            c=y, alpha=.6)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
# It could be interesting to label the different structures here
#                label=list(structure_features.keys())[i])
#    plt.text(min(dat_reduced[:, 0])-.2,max(dat_reduced[:, 1])-.5,'Outlier', color=colors[0])
#    plt.text(min(dat_reduced[:, 0])-.2,max(dat_reduced[:, 1])-1.5,'Inlier', color=colors[1])
    plt.savefig('umap.png')
    plt.show()

def plot_hist(x,bins,feat_names):
    for i in range(x.shape[1]):
        plt.hist(x[feat_names[i]],bins=bins)
        plt.xlabel(feat_names[i])
        plt.savefig('hist_'+feat_names[i]+'.png')
        plt.close() 

def plot_scatter(x,ref,feat_names):
    y=x.pop(ref)
    feat_names.remove(ref)
    if exists('sample_collection.csv'):
        new_x=pd.read_csv('sample_collection.csv')
        new_y=new_x.pop(ref)
    for i in range(x.shape[1]):
        plt.scatter(y,x[feat_names[i]],alpha=.7)
        plt.xlabel(ref)
        plt.ylabel(feat_names[i])
        if exists('sample_collection.csv'):
            plt.scatter(new_y,new_x[feat_names[i]],alpha=.7)
        plt.savefig('scatter_'+feat_names[i]+'_vs_'+ref+'.png')
        plt.close() 

# Plot confusion matrix
def plot_confmat_multi(y_test,y_predict,estimator,unique):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_predict),
        display_labels=unique
    )

    cmp.plot(ax=ax)
    plt.savefig('cf_'+str(estimator)+'.png');
