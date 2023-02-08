import numpy as np 
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV,\
validation_curve,learning_curve,ShuffleSplit,LeaveOneOut,LearningCurveDisplay
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF,DotProduct,ConstantKernel,Matern,RationalQuadratic,ExpSineSquared
from sklearn.metrics import mean_tweedie_deviance, make_scorer,confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd
import math, predict_evaluate, plot

# Function to calculate the learning curve
def l_curve(estim,score,estim_name,params,x,y,best_score,cv):
    lc=LearningCurveDisplay.from_estimator(estim,x,y,cv=cv,shuffle=True,random_state=42,\
    score_type="both",scoring=score,ax=plt.gca(),train_sizes=np.linspace(.1,1,9),\
    line_kw= {"marker": "o"})
    plt.savefig('test.png')
    plt.clf()

    _, axes = plt.subplots(1, 3, figsize=(20, 5))
# Add best_params_ and best_score_ to the figure
    params['SCORE']=round(best_score,3)
    _.suptitle(params)
    axes[0].set_title('Learning curve '+estim_name)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(str(score))
    train_sizes,train_scores,test_scores,fit_times,score_times=learning_curve(
    estim,x,y,scoring=score,train_sizes=np.linspace(.1,1.0,7),\
    cv=cv,return_times=True,shuffle=True,random_state=42)
# Calculate mean and std for each # samples, and mean of whole dataset (last
# item)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
# In case MSE, negate and convert to RMSE    
    if 'neg_mean_squared_error' == score:
        train_scores_mean = np.sqrt(abs(train_scores_mean))
        train_scores_std = np.sqrt(abs(train_scores_std))
        test_scores_mean = np.sqrt(abs(train_scores_mean))
        test_scores_std = np.sqrt(abs(train_scores_std))
        score='RMSE'
    train_sc=train_scores_mean[len(train_scores_mean)-1]
    train_std=train_scores_std[len(train_scores_mean)-1]
    test_sc=test_scores_mean[len(test_scores_mean)-1]
    test_std=test_scores_std[len(test_scores_mean)-1]
# Print train and test scores with full size of database (sometimes hard to read
# from the graph)
    print('     Final scores of',score)
    print('      Training = ',round(train_sc,4), round(train_std,4))
    print('      Test = ',round(test_sc,4),round(test_std,4))
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
        plt.savefig(estim_name+'_power'+str(estim.get_params()['power'])+'_'+str(score)+'_lc.png')
    else:
        plt.savefig(estim_name+'_'+str(score)+'_lc.png')
    plt.clf()
#    plt.show()

def val_curve(estim,score,estim_name,key,x,y,cv):
    print('\n  ~~~ Validation curves is being calculated ~~~')
# Extract list to dict
    param_range= np.logspace(-7, 12, 19)
#        param_range= np.array([0,.01,.02,.04,.08,.1,.3,.7,1,2,5])
#        param_range= np.array(params[key])
    train_scores,test_scores=validation_curve(estim,x,y,param_name=key,
                param_range=param_range,scoring=score,cv=cv)
# Print
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('Validation Curve with '+estim_name)
    plt.xlabel(key)
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
    print('    Saved to '+estim_name+'_'+score+'_vc.png')
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
        short_score,classification,class_dim,cv,unique):
    print('\n  Set with',x.shape[0],'samples and',x.shape[1],'features')
    print('  Training set size: ',x_train.shape[0])
    print('  Test set size: ',x_test.shape[0],'\n')
    print('  ~~~ Tuning of the parameters ~~~')
    print('   Scores performed on test set')
# Set the parameters by cross-validation
    if classification:
        if class_dim == 2:
            scores=['accuracy','balanced_accuracy','precision','recall','roc_auc','precision']
        else:
            scores=['balanced_accuracy','accuracy','f1_weighted',\
             'precision_weighted','recall_weighted','roc_auc_ovr_weighted',\
            'roc_auc_ovo_weighted']
            ref_score='f1_weighted'
#        scores=[None,'accuracy','balanced_accuracy','roc_auc_ovr','neg_log_loss','roc_auc_ovo',\
#        'roc_auc_ovr_weighted','roc_auc_ovo_weighted']
    else:
# If short_score=True, evaluate only scores. If False, also neg
        scores=['explained_variance','max_error','r2',\
                'neg_mean_absolute_error','neg_mean_squared_error',\
                'neg_mean_absolute_percentage_error']
        neg=['neg_mean_absolute_error','neg_mean_squared_error',\
        'neg_root_mean_squared_error','neg_mean_squared_log_error',\
        'neg_median_absolute_error','neg_mean_gamma_deviance','neg_mean_absolute_percentage_error']
# Evaluate all scores
        if not short_score:
            scores=scores+neg
    estim_name=estimator.__class__.__name__
    if estim_name=='SVR' or estim_name=='SVC':
        tuned_parameters = [{'kernel': ['rbf','linear','sigmoid'],'C': [.001,.01,.1,1, 10, 100, 1000]},
                            {'kernel': ['poly'],'C':[.001,.01,.1,1, 10, 100], 'degree': [1,2,3,4,5]}]
    elif estim_name=='GaussianProcessRegressor' or estim_name=='GaussianProcessClassifier':
        rbf=RBF()
        ck=ConstantKernel()
        mat=Matern()
        default=None
        tuned_parameters = [{'kernel':[rbf]}]#'RationalQuadratic','ExpSineSquared','DotProduct']}]
    elif estim_name=='LogisticRegression':
#        tuned_parameters = [{'penalty':['l2'],'solver':['liblinear','newton-cholesky','newton-cg',\
#                'lbfgs','sag','saga'],'class_weight':['balanced',None]},
#        {'penalty':['l1'],'solver':['liblinear','saga'],'class_weight':['balanced',None]},
#        {'penalty':['elasticnet'],'class_weight':['balanced',None]}]
        tuned_parameters = [{'penalty':['l2'],\
                'solver':['liblinear','newton-cholesky','newton-cg','lbfgs','sag','saga'],\
                'class_weight':[None,'balanced'],'C':[.00001,.0001,.001,.01,.1,1,10,100]},\
        {'penalty':['l1'],'solver':['liblinear','saga'],'C':[.00001,.0001,.001,.01,.1,1,10,100],\
                    'class_weight':[None,'balanced']},
        {'penalty':['elasticnet'],'solver':['saga'],'C':[.00001,.0001,.001,.01,.1,1,10,100],\
                    'class_weight':[None,'balanced'],'l1_ratio':[.1,.3,.5,.7,.9]}]
    elif estim_name=='SGDRegressor':
        tuned_parameters = [{'loss':['squared_error','huber','epsilon_insensitive',\
                    'squared_epsilon_insensitive'],\
                    'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1],\
                    'learning_rate': ['constant','optimal','invscaling','adaptive']}]
    elif estim_name=='SGDClassifier':
        tuned_parameters = [{'loss': ['hinge','modified_huber','squared_hinge','perceptron'],\
                        'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1],\
                        'learning_rate': ['constant','optimal','invscaling','adaptive']}]
    elif estim_name=='TweedieRegressor':
        tuned_parameters = [{'power':[0,1,2,3],'alpha': [.001,.005,.01,.05,0.3,.5,.7,.9,1,5,10]}]
    elif estim_name=='GaussianNB':
        tuned_parameters = [{'var_smoothing':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]}]
    elif estim_name=='BernoulliNB':
        tuned_parameters = [{'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1]}]
    elif estim_name=='KNeighborsRegressor' or estim_name=='KNeighborsClassifier':
        tuned_parameters = [{'n_neighbors':[4,5,6,8,10,12,14,16,18],\
        'metric':['cityblock','cosine','minkowsky','euclidean','haversine',\
        'l1','l2','manhattan','nan_euclidean']}]
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
# ETC is incomplete
        elif estim_name=='ExtraTreesClassifier':
            tuned_parameters = [{'max_depth':[None,1,2,3,4,5],'criterion':['gini','entropy'],\
            'min_samples_split':[2,3,4,5,6],'min_samples_leaf':[2,3,4,5,6],\
            'min_weight_fraction_leaf':[0.0,0.2,0.4,0.6,0.8],'max_features':['sqrt','log2',None]}]
        elif estim_name=='DecisionTreeClassifier':
            tuned_parameters = [{'splitter':['best'],'criterion':['gini','entropy'],\
            'class_weight':['balanced',None],'ccp_alpha':[0.0,.5,1.0],\
            'max_features':['auto','sqrt','log2',None]},
            {'splitter': ['random'], 'criterion': ['gini','entropy'],\
            'class_weight':['balanced',None],'ccp_alpha':[0.0,.5,1.0],\
            'max_features':['auto','sqrt','log2',None]}]
    elif estim_name=='MLPRegressor' or estim_name=='MLPClassifier':
        tuned_parameters = [{'activation':['identity','logistic','tanh','relu'],
                    'learning_rate':['constant','invscaling','adaptive'],
            'alpha': [0,.001,.005,.01,.05,.1,.5,1,5,10]}]

# Loop over all scorers defined above
    for score in scores:
        clf = GridSearchCV(estimator, tuned_parameters,
                scoring=score,cv=cv,return_train_score=True)
        clf.fit(x_train, y_train)
# Prepare the dataframe with results, to print the mean and standard deviation
# of the test score for the best set of parameters
        results_df = pd.DataFrame(clf.cv_results_)
        results_df = results_df.sort_values(by=['rank_test_score'])
        results_df = (results_df.set_index(results_df["params"].apply(
            lambda v: "_".join(str(val) for val in v.values()))))
# Print results
        if not classification:
            if score in neg:
                print('\n ·',score,':',-round(clf.score(x_test,y_test),4))
            else:
                print('\n ·',score,':',round(clf.score(x_test,y_test),4))
        else:
            print('\n ·',score,':',round(clf.score(x_test,y_test),4))
        print(results_df[['mean_test_score','std_test_score','mean_train_score','std_train_score']].head(1))
        print(clf.best_estimator_)
# Validation curve only if regularization parameter is defined for estimator
        key=None
        if estim_name == 'MLPClassifier' or estim_name == 'TweedieRegressor'\
        or estim_name == 'MLPRegressor' or estim_name == 'BernoulliNB':
            key='alpha'
        elif estim_name == 'SVC' or estim_name=='LogisticRegressor':
            key='C'
        if key:
            if score == 'neg_mean_squared_error' or score == 'r2'\
            or score == 'f1_weighted':
                val_curve(clf.best_estimator_,score,estim_name,key,x,y,cv)

# Function to print the score with respect to the fold: informs whether the
# results are dependent on the fold
##        covar_print(results_df,score,estim_name)
##        bayesian_test(results_df,x,y)

# Uncomment to print all grid results
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
    print('\n            Summary           ')
    print('            -------           ')
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
        plot.plot_confmat_multi(y_test,y_predict,estim_name,unique)
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
        clf = GridSearchCV(estimator, tuned_parameters,
                scoring=ref_score,cv=cv,return_train_score=True)
        clf.fit(x_train, y_train)
        print('\n  ~~~ Multi-class classification problem ~~~')
        print('\n   - Using',ref_score,'as reference score metric\n')
# Perform predict on the test set
        y_predict=clf.predict(x_test)
# Print the actual set of parameters after CV tuning
        y_test=y_test.to_numpy()
# Plot confusion matrix for multilabel (labels passed in unique variable)
        plot.plot_confmat_multi(y_test,y_predict,estim_name,unique)
        print('   - Confusion matrix plotted to',estim_name+'_cm.png')
# In case it is a regression problem
    elif not classification:
# Learning curve with optimal hyperparameters and for loss/r2 scoring functions
        score_lc=['neg_mean_squared_error','r2']
        for i in score_lc:
            l_curve(clf.best_estimator_,i,estim_name,clf.best_params_,x,y,clf.best_score_,cv)
# Perform CV with all the scoring functions, in order to remove random effects
# on the evaluation of the accuracy of the model
    predict_evaluate.cv_perform(clf.best_estimator_,x,y,cv,scores,classification,class_dim)
# Obsolete: given by cross_validate
# Perform predict on the test set
#        y_predict=best_clf.predict(x_test)
#        y_test=y_test.to_numpy()
##        print('     r2', metrics.r2_score(y_test, y_predict))
##        print('     Max error', metrics.max_error(y_test, y_predict))
##        print('     MAE', metrics.mean_absolute_error(y_test, y_predict))
##        print('     MSE', metrics.mean_squared_error(y_test, y_predict))
##        print('     MA%E', metrics.mean_absolute_percentage_error(y_test, y_predict))
    return clf.best_params_

def trainmod(x,y,feat_names,short_score,classification,estimator,cv):
    estimators_list= []
    class_dim=0
    for i in estimator:
        if classification:
# If it is a classification problem, find out number of classes and counts
            unique, counts=np.unique(y,return_counts=True)
            print('    ',len(unique),'clases')
            print('     Class  Counts')
# Store number of classes in class_dim
            class_dim= len(unique)
            for j in range(class_dim):
                print('      ',unique[j],'    ',counts[j])
            if i == 'knn':
                from sklearn.neighbors import KNeighborsClassifier as KNC
                estimators_list.append(KNC())
            elif i == 'svm':
                estimators_list.append(svm.SVC(probability=True,random_state=42))
            elif i == 'sgd':
                from sklearn.linear_model import SGDClassifier as SGDC
                estimators_list.append(SGDC(max_iter=10000))
            elif i == 'dt':
                from sklearn.tree import DecisionTreeClassifier as DTC
                estimators_list.append(DTC())
            elif i == 'dt':
                from sklearn.ensemble import ExtraTreesClassifier as ETC
                estimators_list.append(ETC())
            elif i == 'gnb':
                from sklearn.naive_bayes import GaussianNB as GNB
                estimators_list.append(GNB())
            elif i == 'bnb':
                from sklearn.naive_bayes import BernoulliNB as BNB
                estimators_list.append(BNB())
            elif i == 'lr':
                from sklearn.linear_model import LogisticRegression as LR
                estimators_list.append(LR(random_state=42))
            elif i == 'linr':
                print('\n  This algorithm is only available for regression tasks.')
                exit()
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
                estimators_list.append(SGDR(max_iter=10000))
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
                estimators_list.append(MLPR(solver='lbfgs'))
# Create StandardScaler instance
#    pp=StandardScaler()
    pp=PowerTransformer()
# Scale x
    x=pp.fit_transform(x)
# Create train and test sets from the original dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.3, random_state=42)
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

    print('\n  Tuning hyperparameters:')
    print('   Scores performed on test set\n')
#    estimators_list.append(TR(power=1))
#    estimators_list.append(TR(power=2,max_iter=50000))
#    estimators_list.append(TR(power=3,max_iter=50000))
    for i in estimators_list:
# Evaluate the models straight away (no cv) to check gross performance
        estim_name=i.__class__.__name__
        print('\n  ------------------------------------')
        print(' ',estim_name,'- Best set found for',':')
        print('  ------------------------------------\n')
        print('  ~~~ Default parameters ~~~')
        i.fit(x_train,y_train)
        print('  ',i)
#        print('   Parameters:',i.get_params())
        print('   Score:',i.score(x_test,y_test))
# Perform cross-validation
        pars=gridsearchcv(i,x_train,y_train,x_test,y_test,x,y,feat_names,\
        short_score,classification,class_dim,cv,unique)
