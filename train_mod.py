import numpy as np 
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, validation_curve, learning_curve,ShuffleSplit
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.linear_model import TweedieRegressor as TR
from sklearn.metrics import mean_tweedie_deviance, make_scorer
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd


def l_curve(estim,score,estim_name,params,x,y):
#    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title('Learning curve '+estim_name)
#    if ylim is not None:
#        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes,train_scores,test_scores,fit_times,score_times=learning_curve(
    estim,x,y,train_sizes=np.linspace(.1,1.0,7),cv=cv,return_times=True,shuffle=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
    label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
    label="Cross-validation score")
    axes[0].legend(loc="best")
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")
    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    if estim_name == 'TweedieRegressor':
        plt.savefig(estim_name+'_power'+str(estim.get_params()['power'])+'_lc.png')
    else:
        plt.savefig(estim_name+'_lc.png')
    plt.clf()
#    plt.show()

def val_curve(estim,score,estim_name,params,x,y):
    print(' - Calculating validation curve for '+estim_name+'\n')
# Extract list to dict
    params=params[0]
    for key in params:
      if key=='alpha':
        param_range= np.logspace(-5, 6, 11)
#        param_range= np.array([0,.01,.02,.04,.08,.1,.3,.7,1,2,5])
#        param_range= np.array(params[key])
        train_scores,test_scores=validation_curve(
        estim,x,y,param_name=key,param_range=param_range)#,scoring=score)
# Print
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title('Validation Curve with '+estim_name)
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Score')
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
#        plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
        plt.fill_between(param_range,test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
        plt.legend(loc="best")
        plt.savefig(estim_name+'_vc.png')
        plt.clf()
#        plt.show()

def corrected_std(diff, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : int
        Variance-corrected standard deviation of the set of differences.
    """
    n = n_train + n_test
    corrected_var = (
        np.var(diff, ddof=1) * ((1 / n) + (n_test / n_train))
    )
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

def bayesian_test(results_df,x,y):
# Extracted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py
# create df of model scores ordered by perfomance
    model_scores = results_df.filter(regex=r'split\d*_test_score')
# create df of model scores ordered by perfomance
    model_scores = results_df.filter(regex=r'split\d*_test_score')
    model_1_scores = model_scores.iloc[0].values  # scores of the best model
    model_2_scores = model_scores.iloc[1].values  # scores of the second-best model
    diff= model_1_scores - model_2_scores
    n=diff.shape[0] # Number of test sets
    df = n-1
    n_train = len(list(KFold().split(x, y))[0][0])
    n_test = len(list(KFold().split(x, y))[0][1])

# intitialize random variable
    t_post = t(
        df, loc=np.mean(diff),
        scale=corrected_std(diff, n_train, n_test)
    )
# Plot the results
    x = np.linspace(t_post.ppf(0.001), t_post.ppf(0.999), 100)

    plt.plot(x, t_post.pdf(x))
    plt.xticks(np.arange(-0.04, 0.06, 0.01))
    plt.fill_between(x, t_post.pdf(x), 0, facecolor='blue', alpha=.2)
    plt.ylabel("Probability density")
    plt.xlabel(r"Mean difference ($\mu$)")
    plt.title("Posterior distribution")
    plt.close('all')
#    plt.show()

    better_prob = 1 - t_post.cdf(0)

    print(' - Bayesian analysis')
    print(f" Probability of {model_scores.index[0]} being more accurate than "
      f"{model_scores.index[1]}: {better_prob:.3f}")
    print(f" Probability of {model_scores.index[1]} being more accurate than "
      f"{model_scores.index[0]}: {1 - better_prob:.3f}")
    rope_interval = [-0.01, 0.01]
    rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])

    print(f" Probability of {model_scores.index[0]} and {model_scores.index[1]} "
      f"being practically equivalent: {rope_prob:.3f}")


def covar_print(results_df,score,estim_name):
# create df of model scores ordered by perfomance
    model_scores = results_df.filter(regex=r'split\d*_test_score')
# plot 30 examples of dependency between cv fold and AUC scores
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:30],
        dashes=False, palette='Set1', marker='o', alpha=.5, ax=ax
    )
    ax.set_xlabel("CV test fold", size=12, labelpad=10)
    ax.set_ylabel("Score", size=12)
    ax.tick_params(bottom=True, labelbottom=False)
    plt.savefig('covar'+estim_name+'_'+score+'.png')
    plt.clf()

# print correlation of AUC scores across folds
    print(f"Correlation of models:\n {model_scores.transpose().corr()}")

def gridsearchcv(estimator,x_train,y_train,x_test,y_test,x,y,feat_names):
    print('\n  Total set size: ',x.shape[0])
    print('  Training set size: ',x_train.shape[0])
    print('  Test set size: ',x_test.shape[0],'\n')
# Set the parameters by cross-validation
#    tweedie_deviance=make_scorer(mean_tweedie_deviance,power=2)
    scores=['explained_variance','r2','neg_mean_squared_error','neg_mean_absolute_error']
    estim_name=estimator.__class__.__name__
#    if estim_name=='LogisticRegression':
#        tuned_parameters = [{'penalty': ['none'], 'C': [.05,0.3,.5,.7,.9,1,5,10]},
#        {'penalty': ['l2'], 'C': [.05,0.3,.5,.7,.9,1,5,10]},
#        {'penalty': ['l1'], 'C': [.05,0.3,.5,.7,.9,1,5,10]},
#        {'penalty': ['elasticnet'], 'C': [.05,0.3,.5,.7,.9,1,5,10]}]
    if estim_name=='SVR':
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale', 'auto'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'gamma': ['scale', 'auto'], 'C': [.1,1, 10, 100]},
                    {'kernel': ['poly'], 'gamma': ['scale', 'auto'], 'C': [.1,1, 10, 100]},
                    {'kernel': ['sigmoid'], 'gamma': ['scale', 'auto'], 'C': [.1,1, 10, 100]}]
    elif estim_name=='SGDRegressor':
        tuned_parameters = [{'loss': ['squared_loss'], 'alpha': [1e-5, 1e-4,
            1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['huber'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1], 
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['epsilon_insensitive'], 'alpha': [1e-5, 1e-4,
                        1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['squared_epsilon_insensitive'], 'alpha': [1e-5,
                        1e-4, 1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']}]
    elif estim_name=='TweedieRegressor':
        tuned_parameters = [{'alpha': [.05,0.3,.5,.7,.9,1,5,10]}]
#        {'power': [1], 'link':['log'],'alpha': [0.3,.5,.7,.9,1,5,10]},
#        {'power': [2], 'alpha': [0.3,.5,.7,.9,1,5,10]},
#        {'power': [3], 'alpha': [0.3,.5,.7,.9,1,5,10]}]
    elif estim_name=='KNeighborsRegressor':
         tuned_parameters = [{'algorithm': ['auto'], 'p': [1,2,7], 'n_neighbors': [1,2,5,7,11]}]
#        {'algorithm': ['ball_tree'],'p': [1,2,7], 'n_neighbors': [2,5,7,11]},
#        {'algorithm': ['kd_tree'], 'p': [1,2,7], 'n_neighbors': [2,5,7,11]},
#        {'algorithm': ['auto'], 'p': [1,2,7], 'n_neighbors': [2,5,7,11]}]
    elif estim_name=='DecisionTreeRegressor':
# Evaluate, in case of Decision Tree, the feature importances
        if feat_names != False:
            dtr = estimator.fit(x_train,y_train)
            importances = dtr.feature_importances_
            dtr_importances = pd.Series(importances, index=feat_names)
            dtr_importances.plot.bar()
            lab_site=[]
            lab=[]
            for i in range(len(importances)):
                if importances[i] > 0.01:
                    lab_site.append(i)
                    lab.append(feat_names[i])
            plt.ylabel("Mean decrease in impurity")
            plt.xticks(lab_site,lab,rotation=0)
            plt.savefig('dtr_feat_importances.png')
            plt.close()
        tuned_parameters = [{'splitter': ['best'],
            'criterion': ['mse','friedman_mse','mae','poisson']},
        {'splitter': ['random'],
            'criterion': ['mse','friedman_mse','mae','poisson']}]
    elif estim_name=='MLPRegressor':
        tuned_parameters = [
        {'solver':['lbfgs'],
            'alpha': [0,.001,.005,.01,.05,.1,.5,1,5,10]},
        {'solver':['adam'],
            'alpha': [0,.001,.005,.01,.05,.1,.5,1,5,10]}]

    print(' ',estim_name,'- Best set found for',':')
    print('  ---------------------------------')
    for score in scores:
# Declare the GridSearchCV strategy
        clf = GridSearchCV(estimator, tuned_parameters, scoring=score)
        clf.fit(x_train, y_train)
# Print results
        print('\n   ~',score,'~')
# Prepare the dataframe with results, to print the mean and standard deviation
# of the test score for the best set of parameters
        results_df = pd.DataFrame(clf.cv_results_)
        results_df = results_df.sort_values(by=['rank_test_score'])
        results_df = (results_df.set_index(results_df["params"].apply(
            lambda v: "_".join(str(val) for val in v.values()))))
        print(results_df[['mean_test_score','std_test_score']].head(1))
        print('Scoring function applied to test set of dimension:',
                round(clf.score(x_test,y_test),4),'\n')
#        print(clf.predict(x_test), y_test)
# Validation curve
        if estim_name=='MLPRegressor':
            val_curve(estimator,score,estim_name,tuned_parameters,x,y)
# Learning curve
        l_curve(estimator,score,estim_name,tuned_parameters,x,y)

# Function to print the score with respect to the fold: informs whether the
# results are dependent on the fold
##        covar_print(results_df,score,estim_name)
        bayesian_test(results_df,x,y)

# Uncomment to print all grid results
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
    print()
    return clf.best_params_

def trainmod(x,y,feat_names):
# Create StandardScaler instance
    sscaler=StandardScaler()
# Create train and test sets from the original dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33, random_state=42)
# Perform standard scaling; fit only on the training set
    x_train=sscaler.fit_transform(x_train)
# Scale x
    x=sscaler.fit_transform(x)
# Trasform both training and test
    x_test=sscaler.transform(x_test)
# Create instance of PCA, requesting to reach a 95% of explained variance
    pca=PCA(.9999)
# Fit training set only
    x_train=pca.fit_transform(x_train)
# Transform both training and test sets
    x_test=pca.transform(x_test)
    print('\n-> Applied feature reduction PCA')
    print('  Reduced from ',x.shape[1],' to ',x_train.shape[1],' features')

    print('\n-> Tuning hyperparameters:')
    estimators_list= []
    estimators_list.append(SGDR())
    estimators_list.append(svm.SVR())
    estimators_list.append(KNR())
    estimators_list.append(DTR(random_state=42))
    estimators_list.append(MLPR(random_state=42,max_iter=10000))
    estimators_list.append(TR(power=0,max_iter=20000,alpha=.01))
#    estimators_list.append(TR(power=1,max_iter=20000))
#    estimators_list.append(TR(power=2,max_iter=50000))
#    estimators_list.append(TR(power=3,max_iter=50000))
    for i in estimators_list:
        pars=gridsearchcv(i,x_train,y_train,x_test,y_test,x,y,feat_names)

# Create instance of model SVM
#    svr=svm.SVR()
# Fit on training set
#    svr.fit(x_train,y_train)
#    y_pred=svr.predict(x_test)
#    scores=cross_val_score(svr, x_train, y_train, cv=5)
#    print(scores)
#    print('Mean  Std')
#    print(round(scores.mean(),2), round(scores.std(),2))
