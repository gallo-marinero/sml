import pandas as pd
from os.path import exists
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_pca_correlation_graph
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,\
roc_curve,auc

small = 8
medium = 13
large = 20

plt.rc('font', size=large)          # controls default text sizes
plt.rc('axes', titlesize=large)     # fontsize of the axes title
plt.rc('axes', labelsize=large)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=large)    # fontsize of the tick labels
plt.rc('ytick', labelsize=large)    # fontsize of the tick labels
plt.rc('legend', fontsize=large)    # legend fontsize
plt.rc('figure', titlesize=large)  # fontsize of the figure title
plt.rc('figure', labelsize=large)  # fontsize of the figure title

def importance_bar(feat_names,importance):
    print('      Feature      Score')
    for i,v in enumerate(importance):
        print(' %15s   %.5f' % (feat_names[i],v))
# plot feature importance
    fig, ax = plt.subplots()

    y_pos=np.arange(len(feat_names))
    hbars = ax.barh(y_pos, importance, align='center')
    ax.set_yticks(y_pos, labels=feat_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
#    ax.set_title('How fast do you want to go today?')

# Label with specially formatted floats
    ax.bar_label(hbars, fmt='%.2f')
    ax.set_xlim(left=min(importance)-6,right=max(importance)+6)  # adjust xlim to fit labels
    plt.savefig('importance.png',bbox_inches='tight')


def correlation_circle(x,feat_names,estimator):
    print('\n -> Plotting correlation circle to correlation_circle.png')
    figure, correlation_matrix = plot_pca_correlation_graph(x, feat_names,\
                                dimensions=(1, 2),figure_axis_size=10)
    plt.title('')
    plt.savefig('correlation_circle.png')
    plt.savefig('correlation_circle.pdf')
    plt.close()
    plt.clf()

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

'''
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
'''

def plot_hist(x,bins,feat_names,scal):
    for i in range(x.shape[1]):
        plt.hist(x[feat_names[i]],bins=bins)
        plt.xlabel(feat_names[i])
        plt.ylabel('# samples')
        if scal:
            plt.savefig('hist_'+feat_names[i]+'_'+scal+'.png')
        else:
            plt.savefig('hist_'+feat_names[i]+'.png')
        plt.clf() 
        plt.close() 
        plt.cla() 

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

def plot_roc(y_test,y_predict,unique,estimator):
    fpr,tpr,threshold=roc_curve(y_test,y_predict,pos_label='High')
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.axis("square")
    plt.xlabel('False'+unique[0]+' Rate')
    plt.ylabel('True'+unique[1]+'  Rate')
    plt.title('One-vs-Rest ROC curve:\nHigh vs (Setosa & Versicolor)')
    plt.legend()
    plt.savefig('roc_'+str(estimator)+'.png')
