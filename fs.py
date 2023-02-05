from sklearn.model_selection import train_test_split,RepeatedKFold,cross_val_score,RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE, RFECV
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import TweedieRegressor as TR
from numpy import mean, std
import predict_evaluate, train
import numpy as np

def vt(x,feat_names):
# Remove features that are 0/1 in more than 80% of the samples
    print('\n ~ Remove features that are 0/1 in > 80% of the samples ~')
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit(x)
    dropped=[]
    selected=sel.get_support(indices=False)
    for i in range(len(feat_names)):
        if not selected[i]:
            dropped.append(feat_names[i])
    print(' Removed features:')
    for i in dropped:
        print('  ',i)
    return dropped

def plot_skb(x_train,y_train,x_test):
    print('Bar plots saved for Mutual information and F-regression.')
    for i in range(2):
# Configure to select all features
        if i==0:
            title='Mutual_information'
            fs = SelectKBest(score_func=mutual_info_regression, k='all')
        elif i==1:
            title='F_regression'
            fs = SelectKBest(score_func=f_regression, k='all')
# Learn relationship from training data
        fs.fit(x_train, y_train)
# Transform train input data
        x_train_fs = fs.transform(x_train)
# Transform test input data
        x_test_fs = fs.transform(x_test)
# Plot the scores
        plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
        plt.xlabel('Feature index')
        plt.ylabel(title)
#        plt.show()
        plt.savefig(title+'_mfs.png')
        plt.clf()

def skb_bestk(x,y,k,model,score_f,feat_names,short_score,classification,estimator,cv):
    print('\n Best configuration with',k,'features:')
    print('   ID     Name        Score')
    skb = SelectKBest(score_func=score_f,k=k)
    skb = skb.fit(x,y)
    x_skb = skb.transform(x)
    scores=np.argsort(skb.scores_)
    for i in range(len(skb.scores_)-1,len(skb.scores_)-k-1,-1):
        print('   {0:2} {1:13s} {2:7.3f}'.format(scores[i],feat_names[scores[i]],skb.scores_[scores[i]]))
    print('\n   Training with',k,'features:')
    train.trainmod(x_skb,y,feat_names,short_score,classification,estimator,cv)
            

def skb(x,y,feat_names,short_score,classification,estimator,cv):
# Define the evaluation method and model
    model=TR(power=0,max_iter=2000)
    for i in range(3):
        if i==0:
            print('\n')
            print('\n ~ Fit with all features ~ ')
# Evaluate the model with all features in predict_evaluate.py module
            x_train,y_train,x_test,y_test,eval_model=predict_evaluate.pred_eval(x,y,model)
##            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,
 ##           random_state=42)
# Fit the data to a Linear regression
  ##          model.fit(x_train,y_train)
# Evaluate the model
   ##         eval_model=model.predict(x_test)
# Evaluate the prediction
            mae=mean_absolute_error(y_test, eval_model)
            print('# of features considered: ',x_train.shape[1])
            print('Mean Absolute Error: {:5.3f}'.format(mae))
# Plot the score for each feature
            plot_skb(x_train,y_train,x_test)

# Configure to select all features
        elif i !=0:
            print('\n')
            if i == 1:
                print('\n ~ Features according to mutual information ~ ')
                score_f=mutual_info_regression
            elif i == 2:
                print('\n ~ Features according to F regression ~ ')
                score_f=f_regression
            fs = SelectKBest(score_func=score_f)
            pipeline = Pipeline(steps=[('sel',fs), ('model', model)])
# Define the grid: explore from 2 features to all, in steps of 2
            grid = dict()
            grid['sel__k'] = [i for i in range(2, x.shape[1]+1,2)]
# Define the grid search: use MAE as scoring function
            search = GridSearchCV(pipeline, grid, scoring='r2', cv=cv)
# Perform the search
            results = search.fit(x, y)
# What are scores for the features
#            for i in range(len(search.scores_)):
#                print('{:15.13s} {:6.4f}'.format(x.columns[i], search.scores_[i]))
# Plot the scores
#            pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
#            pyplot.show()
            # summarize best
            print(' Mean Absolute Error: {:5.3f}'.format(results.best_score_))
            print(' Best configuration: {} features'.format(results.best_params_['sel__k']))
# Summarize all
            means = results.cv_results_['mean_test_score']
            params = results.cv_results_['params']
            for mean, param in zip(means, params):
                print("   %.3f with: %r" % (mean, param['sel__k']))
            
            skb_bestk(x,y,results.best_params_['sel__k'],model,score_f,feat_names,short_score,classification,estimator,cv)

def rfecv(x,y,feat_names,short_score,classification,estimator,cv):
    print('\n\n ~ Recursive feature elimination with cross-validation ~ ')
    model = TR(max_iter=2000)
    rfecv = RFECV(estimator=model,cv=cv)#,n_features_to_select=i)
    rfecv = rfecv.fit(x,y)
    x_rfecv = rfecv.transform(x)
    for i in range(len(rfecv.support_)):
        if rfecv.support_[i]:
            print('   {0:2} {1:13s} {2:7.3f}'.format(i,feat_names[i],rfecv.cv_results_['mean_test_score'][i]))

    print('\n   Training with',rfecv.n_features_,'features:')
    train.trainmod(x_rfecv,y,feat_names,short_score,classification,estimator,cv)
#    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
#    n_scores = cross_val_score(pipeline, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
#    print('  %i   %.3f   (%.3f)' % (rfe.n_features_, mean(n_scores), std(n_scores)))
