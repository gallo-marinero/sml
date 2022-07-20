# Module for preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Add polynomial features to the dataset of degree 'poly_feat' (defined by user)
def polynomial_features(x,poly_feat):
    print('\n -> Polynomial features transformation')
    print('       Degree ', poly_feat)
# Get original number of features for information
    orig_feats=np.shape(x)[1]
    poly=PolynomialFeatures(poly_feat)
    x=poly.fit_transform(x)
# Print new number of features
    print('       Training set transformed from',orig_feats,'to',np.shape(x)[1],'features')
# Label columns in pandas dataframe format
    x=pd.DataFrame(x,columns=poly.get_feature_names_out())
    return x
