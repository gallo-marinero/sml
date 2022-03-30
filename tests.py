import numpy as np
import seaborn as sns
import pandas as pd
from scipy.spatial import distance 
from scipy import stats 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.datasets import make_classification

def rf_diff_dist(x,y,yname):
# The idea is pretty simple: build a random forest model (or any other
# classifier) whose goal is to classify a datapoint in either “training” or
# “test”. You shouldn’t be able to correctly decide whether a row belongs to the
# training or test set, they should be indistinguishable. So, if our model
# performs too well, then you can blame the low test score on the test set
# having a different distribution from the training set.
    print('\n ~ Random Forest test ~ ')
    for i in range(2):
        if i == 0:
            print('    Nonrandom train/test split: ')
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,shuffle=False)
        elif i == 1:
            print('\n    Random train/test split (the ability of RF to classify into groups should drop)')
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,shuffle=True)
# We annotate the training data with the 'train' label in a new column
        x_train = x_train.assign(dataset='train')
#...and add back the target variable. It's natural to use it in the random forest classifier
        x_train = pd.concat([x_train, y_train], axis = 1)
# We do the same with the test data
        x_test = x_test.assign(dataset='test')
        x_test = pd.concat([x_test, y_test], axis = 1)
# We add everything together
        x_rf = pd.concat([x_train, x_test])
# We encode the old target variable for the classifier
#        x_rf = pd.get_dummies(x_rf, columns=[yname])
# The dataset column - that is, whether the datapoint belongs to the training or the test dataset - is exactly what we are trying to predict
        y_rf = x_rf.pop('dataset')
# The following is just a good and old Random Forest classifier
        if i == 0:
            x_rf_train, x_rf_test, y_rf_train, y_rf_test = train_test_split(
     x_rf, y_rf,test_size=.33,random_state=42)
        elif i == 1:
            x_rf_train, x_rf_test, y_rf_train, y_rf_test = train_test_split(
            x_rf, y_rf,test_size=.33,random_state=42,stratify = y_rf)

        clf = RF(max_depth=2, random_state=42)
        clf.fit(x_rf_train.values, y_rf_train.values)

        print(clf.score(x_rf_test.values, y_rf_test.values))

def ks(x,y):
# Perform Kolmogorov-Smirnov test to check whether the distribution of test and
# train sets is the same or not.
# Null hypothesis: distributions are identical
# If shuffle=False:
#   The distributions might be very different (large statistic value)
# If shuffle=True:
#   The distributions should be not very different (low statistic value) and the
#   pvalue large (not possible to discard the null hypothesis, i.e.: they are
#   identical)
# Split nonrandomly
    print('\n ~ Kolmogorov-Smirnov test ~ the bigger KS, the more different the distributions')
    print('\n   Null hypothesis: distributions are identical')
    print('      - Nonrandom:')
    print('         The distributions might be very different (large KS)')
    print('      - Random:')
    print('         The distributions should not be very different (low KS, large t)')
    for i in range(2):
        if i == 0:
            sc_nrandom=[]
            p_nrandom=[]
            x_train, x_test, y_train, y_test = train_test_split(
                    x,y,test_size=.33,shuffle=False,random_state=42)
            for j in x.columns:
                sc_nrandom.append(stats.ks_2samp(x_train[j], x_test[j])[0])
                p_nrandom.append(stats.ks_2samp(x_train[j], x_test[j])[1])
        elif i == 1:
            sc_random=[]
            p_random=[]
            x_train, x_test, y_train, y_test = train_test_split(
                    x,y,test_size=.33,shuffle=True,random_state=42)
            for j in x.columns:
                sc_random.append(stats.ks_2samp(x_train[j], x_test[j])[0])
                p_random.append(stats.ks_2samp(x_train[j], x_test[j])[1])
    print('\n                    Nonrandom          Random train/test split')
    print('\n                   KS    t-value        KS     t-value')
    for i in range(len(sc_nrandom)):
        print('{:17.15} {:7.4} {:7.4}     {:7.4} {:7.4}'.format(
            x.columns[i],float(sc_nrandom[i]),float(p_nrandom[i]),
                float(sc_random[i]),float(p_random[i])))

def mahalanobis(x,y):
    print('\n ~ Mahalanobis distance test ~')
    z=x+y
# We get the covariance matrix of Z
    cov_z = np.cov(z, rowvar = False)
# And get the inverse of the covariance matrix because that's the actual input
# for the Mahalanobis distance
    cov_z_inverse = np.linalg.inv(cov_z)
# Now we apply the mahalanobis distance to each pair of vectors of X. This gets
# us random variables, that is, numbers (dimension = 1)
#    x_1d = [distance.mahalanobis(x1, x2, cov_z_inverse) for x1 in x for x2 in x]

# and the same for Y
    y_1d = [distance.mahalanobis(y1, y2, cov_z_inverse) for y1 in y for y2 in y]

# And finally we apply the Kolmogorov-Smirnov to these new distributions of
# numbers X_1d, Y_1d
#    print(stats.ks_2samp(x_1d, y_1d))
