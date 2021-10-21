import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Perform PCA component analysis
def pca_feat(x,crit,v):
# If an integer is given, perform PCA for crit number of components
    if isinstance(crit,int):
        print('\n-> Performing PCA analysis with',crit,'components')
# If a float is given, perform PCA until crit variance ratio is achieved
    elif isinstance(crit,float):
        print('\n-> Performing PCA analysis until',crit*100,'% of variance is explained')
    pca=PCA(crit)
    x_pca=pca.fit(x).transform(x)
    if v:
        #print(pca.components_)
        print('\nThe',pca.n_components_,'components explain',
                round(100*sum(pca.explained_variance_ratio_),2),'% of the variance')
        for i in range(pca.n_components_):
            print(' ',i+1,round(100*pca.explained_variance_ratio_[i],2))

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

