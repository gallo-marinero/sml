# Module for preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

# Add polynomial features to the dataset of degree 'poly_feat' (defined by user)
def polynomial_features(x,degree):
    print('\n -> Polynomial features transformation')
    print('       Degree ', degree)
# Get original number of features for information
    orig_feats=np.shape(x)[1]
    poly=PolynomialFeatures(degree=degree)
    x=poly.fit_transform(x)
# Print new number of features
    print('       Training set transformed from',orig_feats,'to',np.shape(x)[1],'features')
# Label columns in pandas dataframe format
    x=pd.DataFrame(x,columns=poly.get_feature_names_out())
    return x

# Add polynomial features to the dataset of degree 'poly_feat' (defined by user)
def spline_features(x,degree,knots):
    print('\n -> Spline features transformation')
    print('       Degree ', degree,'; knots',knots)
# Get original number of features for information
    orig_feats=np.shape(x)[1]
    spline=SplineTransformer(degree=degree,n_knots=knots)
    x=spline.fit_transform(x)
# Print new number of features
    print('       Training set transformed from',orig_feats,'to',np.shape(x)[1],'features')
# Label columns in pandas dataframe format
    x=pd.DataFrame(x,columns=spline.get_feature_names_out())
    return x
