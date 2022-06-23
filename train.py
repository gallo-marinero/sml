import numpy as np 
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, validation_curve, learning_curve,ShuffleSplit
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF,DotProduct,ConstantKernel,Matern,RationalQuadratic,ExpSineSquared
from sklearn.metrics import mean_tweedie_deviance, make_scorer,confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd


def l_curve(estim,score,estim_name,params,x,y,best_score):
#    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
# Add best_params_ and best_score_ to the figure
    params['SCORE']=round(best_score,3)
    _.suptitle(params)
    axes[0].set_title('Learning curve '+estim_name)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(str(score))

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
    axes[2].set_ylabel(str(score))
    axes[2].set_title("Performance of the model")
    if estim_name == 'TweedieRegressor':
        plt.savefig(estim_name+'_power'+str(estim.get_params()['power'])+'_lc.png')
    else:
        plt.savefig(estim_name+'_'+str(score)+'_lc.png')
    plt.clf()
#    plt.show()

def val_curve(estim,score,estim_name,params,x,y):
# Extract list to dict
    params=params[0]
    for key in params:
      if key=='alpha':
        param_range= np.logspace(-5, 6, 11)
#        param_range= np.array([0,.01,.02,.04,.08,.1,.3,.7,1,2,5])
#        param_range= np.array(params[key])
        train_scores,test_scores=validation_curve(estim,x,y,param_name=key,
                param_range=param_range,scoring=score)
# Print
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title('Validation Curve with '+estim_name)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(score)
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
        plt.savefig(estim_name+'_'+score+'_vc.png')
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

    print('   - Bayesian analysis')
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

def gridsearchcv(estimator,x_train,y_train,x_test,y_test,x,y,feat_names,\
        short_score,classification,class_dim):
    print('\n  Set with',x.shape[0],'samples and ',x.shape[1],'features')
    print('  Training set size: ',x_train.shape[0])
    print('  Test set size: ',x_test.shape[0],'\n')
    print('  ~~~ Tuning of the parameters ~~~')
    print('   Validation curves calculated for each scoring function')
# Set the parameters by cross-validation
    if classification:
        if class_dim == 2:
            scores=[None,'accuracy','balanced_accuracy','precision','recall','roc_auc','precision']
        else:
            scores=[None,'balanced_accuracy','accuracy']
#        scores=[None,'accuracy','balanced_accuracy','roc_auc_ovr','neg_log_loss','roc_auc_ovo',\
#        'roc_auc_ovr_weighted','roc_auc_ovo_weighted']
    else:
# If short_score=True, evaluate only nonneg scores
        scores=[None,'explained_variance','max_error','r2']
        if not short_score:
            neg=['neg_mean_absolute_error','neg_mean_squared_error',\
        'neg_root_mean_squared_error','neg_mean_squared_log_error',\
        'neg_median_absolute_error','neg_mean_gamma_deviance','neg_mean_absolute_percentage_error']
# Evaluate all scores
            scores=scores+neg
    estim_name=estimator.__class__.__name__
    if estim_name=='SVR' or estim_name=='SVC':
        tuned_parameters = [{'kernel': ['rbf'],'gamma': ['scale', 'auto'], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'],'gamma': ['scale', 'auto'], 'C': [.1,1, 10, 100]},
                            {'kernel': ['poly'],'gamma': ['scale', 'auto'], 'C': [.1,1, 10, 100],
                             'degree': [1,2,3,4,5]},
                            {'kernel': ['sigmoid'], 'gamma': ['scale', 'auto'], 'C': [.1,1, 10, 100]}]
    elif estim_name=='GaussianProcessRegressor' or estim_name=='GaussianProcessClassifier':
        rbf=RBF()
        ck=ConstantKernel()
        mat=Matern()
        default=None
        tuned_parameters = [{'kernel':[rbf]}]#'RationalQuadratic','ExpSineSquared','DotProduct']}]
    elif estim_name=='LogisticRegression':
        tuned_parameters = [{'penalty':['none'],'solver':['newton-cg','lbfgs','sag','saga']},
        {'penalty':['l2'],'solver':['newton-cg','lbfgs','sag','saga'],'C': [.1,.3,.5,1,10,100]},
        {'penalty':['l1'],'solver':['saga'],'C': [.1,.3,.5,1,10,100]},
        {'penalty':['elasticnet'],'solver':['saga'],'C': [.1,.3,.5,1,10,100],'l1_ratio':[.1,.3,.5,.7,.9]}]
    elif estim_name=='SGDRegressor':
        tuned_parameters = [{'loss': ['squared_error'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['huber'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1], 
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['epsilon_insensitive'], 'alpha': [1e-5, 1e-4,
                        1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['squared_epsilon_insensitive'], 'alpha': [1e-5,
                        1e-4, 1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']}]
    elif estim_name=='SGDClassifier':
        tuned_parameters = [{'loss': ['hinge'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['modified_huber'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1], 
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['squared_hinge'], 'alpha': [1e-5, 1e-4,
                        1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']},
                    {'loss': ['perceptron'], 'alpha': [1e-5,
                        1e-4, 1e-3, 1e-2,.1,1],
                        'learning_rate': ['constant','optimal','invscaling','adaptive']}]
    elif estim_name=='TweedieRegressor':
        tuned_parameters = [{'alpha': [.001,.005,.01,.05,0.3,.5,.7,.9,1,5,10]}]
#        {'power': [1], 'link':['log'],'alpha': [0.3,.5,.7,.9,1,5,10]},
#        {'power': [2], 'alpha': [0.3,.5,.7,.9,1,5,10]},
#        {'power': [3], 'alpha': [0.3,.5,.7,.9,1,5,10]}]
    elif estim_name=='KNeighborsRegressor' or estim_name=='KNeighborsClassifier':
        tuned_parameters = [{'weights':['uniform','distance'],'p':[1,2,7],\
                'n_neighbors':[2,4,6,8,10]}]
    elif estim_name=='DecisionTreeRegressor' or estim_name=='DecisionTreeClassifier':
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
        if estim_name=='DecisionTreeRegressor':
            tuned_parameters = [{'splitter': ['best'],
            'criterion': ['squared_error','friedman_mse','absolute_error','poisson']},
        {'splitter': ['random'],
            'criterion': ['squared_error','friedman_mse','absolute_error','poisson']}]
        elif estim_name=='DecisionTreeClassifier':
            tuned_parameters = [{'splitter':['best'],'criterion':['gini','entropy'],\
            'class_weight':['balanced',None],'ccp_alpha':[0.0,.5,1.0],\
            'max_features':['auto','sqrt','log2',None]},
            {'splitter': ['random'], 'criterion': ['gini','entropy'],\
            'class_weight':['balanced',None],'ccp_alpha':[0.0,.5,1.0],\
            'max_features':['auto','sqrt','log2',None]}]
    elif estim_name=='MLPRegressor' or estim_name=='MLPClassifier':
        tuned_parameters = [
        {'solver':['lbfgs'],
            'alpha': [0,.001,.005,.01,.05,.1,.5,1,5,10]},
        {'solver':['adam'],
            'alpha': [0,.001,.005,.01,.05,.1,.5,1,5,10]}]

    for score in scores:
# Declare the GridSearchCV strategy
        clf = GridSearchCV(estimator, tuned_parameters, scoring=score)
        clf.fit(x_train, y_train)
# Prepare the dataframe with results, to print the mean and standard deviation
# of the test score for the best set of parameters
        results_df = pd.DataFrame(clf.cv_results_)
        results_df = results_df.sort_values(by=['rank_test_score'])
        results_df = (results_df.set_index(results_df["params"].apply(
            lambda v: "_".join(str(val) for val in v.values()))))
# Print results
        if not classification:
            if not short_score and score in neg:
                print('\n ·',score,':',-round(clf.score(x_test,y_test),4))
        else:
            print('\n ·',score,':',round(clf.score(x_test,y_test),4))
        print(results_df[['mean_test_score','std_test_score']].head(1))
        print(clf.best_estimator_)
# Validation curve
#        if estim_name=='TweedieRegressor':
#            val_curve(clf.best_estimator_,score,estim_name,clf.best_params_,x,y)
# Learning curve
        l_curve(clf.best_estimator_,score,estim_name,clf.best_params_,x,y,clf.best_score_)

# Function to print the score with respect to the fold: informs whether the
# results are dependent on the fold
##        covar_print(results_df,score,estim_name)
##        bayesian_test(results_df,x,y)

# Uncomment to print all grid results
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
    if class_dim == 2:
        print('\n  ~~~ Binary classification problem ~~~')
        y_predict=clf.predict(x_test)
        print(clf.best_estimator_)
        y_test=y_test.to_numpy()
        print('Accuracy', metrics.accuracy_score(y_test, y_predict))
        print('Precision', metrics.precision_score(y_test, y_predict))
        print('Recall', metrics.recall_score(y_test, y_predict))
        print('ROC_AUC', metrics.roc_auc_score(y_test, y_predict))
        cm=confusion_matrix(y_test, y_predict)
        tn,fp,fn,tp=cm.ravel()
        print('  Confusion matrix plotted to',estim_name+'_cm.png')
        print('    True negatives',tn)
        print('    False positives',fp)
        print('    False negatives',fn)
        print('    True positives',tp)
# If binary classification problem, plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
        disp.plot()
        plt.savefig(estim_name+'_cm.png')
        print()
    elif class_dim > 2:
        print('\n  ~~~ Multi-class classification problem ~~~')
# Perform predict on the test set
        y_predict=clf.predict(x_test)
# Print the actual set of parameters after CV tuning
        print(clf.best_estimator_)
        y_test=y_test.to_numpy()
        print('Accuracy', metrics.accuracy_score(y_test, y_predict))
    return clf.best_params_

def trainmod(x,y,feat_names,short_score,classification,estimator):
    estimators_list= []
    class_dim=0
    for i in estimator:
        if classification:
# If it is a classification problem, find out number of classes and counts
            unique, counts=np.unique(y,return_counts=True)
            print('  ',len(unique),'clases')
            print('  Class  Counts')
# Store number of classes in class_dim
            class_dim= len(unique)
            for j in range(class_dim):
                print('   ',unique[j],'  ',counts[j])
            if i == 'knn':
                from sklearn.neighbors import KNeighborsClassifier as KNC
                estimators_list.append(KNC(n_neighbors=4))
            elif i == 'svm':
                estimators_list.append(svm.SVC())
            elif i == 'sgd':
                from sklearn.linear_model import SGDClassifier as SGDC
                estimators_list.append(SGDC(max_iter=10000))
            elif i == 'dt':
                from sklearn.tree import DecisionTreeClassifier as DTC
                estimators_list.append(DTC())
            elif i == 'lr':
                from sklearn.linear_model import LogisticRegression as LR
                estimators_list.append(LR(max_iter=20000))
            elif i == 'gp':
                from sklearn.gaussian_process import GaussianProcessClassifier as GPC
                estimators_list.append(GPC(n_restarts_optimizer=0))
            elif i == 'nn':
                from sklearn.neural_network import MLPClassifier as MLPC
                estimators_list.append(MLPC(max_iter=10000))
        else:
            if i == 'knn':
                from sklearn.neighbors import KNeighborsRegressor as KNR
                estimators_list.append(KNR())
            elif i == 'svm':
                estimators_list.append(svm.SVR())
            elif i == 'sgd':
                from sklearn.linear_model import SGDRegressor as SGDR
                estimators_list.append(SGDR())
            elif i == 'dt':
                from sklearn.tree import DecisionTreeRegressor as DTR
                estimators_list.append(DTR())
            elif i == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor as GPR
                estimators_list.append(GPR(n_restarts_optimizer=0))
            elif i == 'linr':
                from sklearn.linear_model import TweedieRegressor as TR
                estimators_list.append(TR(power=0))
            elif i == 'nn':
                from sklearn.neural_network import MLPRegressor as MLPR
                estimators_list.append(MLPR(max_iter=10000))
# Create StandardScaler instance
    sscaler=StandardScaler()
# Scale x
    x=sscaler.fit_transform(x)
# Create train and test sets from the original dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.2, random_state=42)
# Perform standard scaling; fit only on the training set
##    x_train=sscaler.fit_transform(x_train)
# Scale x
##    x=sscaler.fit_transform(x)
# Trasform both training and test
##    x_test=sscaler.transform(x_test)
# Create instance of PCA, requesting to reach a 95% of explained variance
##    pca=PCA(.9999)
# Fit training set only
##    x_train=pca.fit_transform(x_train)
# Transform both training and test sets
##    x_test=pca.transform(x_test)
#    print('\n-> Applied feature reduction PCA')
#    print('  Reduced from ',x.shape[1],' to ',x_train.shape[1],' features')

    print('\n  Tuning hyperparameters:\n')
#    estimators_list.append(TR(power=1))
#    estimators_list.append(TR(power=2,max_iter=50000))
#    estimators_list.append(TR(power=3,max_iter=50000))
    for i in estimators_list:
# Evaluate the models straight away (no cv) to check gross performance
        estim_name=i.__class__.__name__
        print('  ------------------------------------')
        print(' ',estim_name,'- Best set found for',':')
        print('  ------------------------------------\n')
        print('  ~~~ Default parameters ~~~')
        i.fit(x_train,y_train)
        print(' ',i)
#        print('   Parameters:',i.get_params())
        print('   Score:',i.score(x_test,y_test))
# Perform cross-validation
        pars=gridsearchcv(i,x_train,y_train,x_test,y_test,x,y,feat_names,short_score,classification,class_dim)
