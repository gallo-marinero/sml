import numpy as np 
from scipy.stats import t
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV,\
validation_curve,learning_curve,ShuffleSplit,LeaveOneOut,LearningCurveDisplay,cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF,DotProduct,ConstantKernel,Matern,RationalQuadratic,ExpSineSquared
from sklearn.metrics import mean_tweedie_deviance,\
make_scorer,confusion_matrix,ConfusionMatrixDisplay,precision_score,f1_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd
import math, predict_evaluate, plot, joblib, load_predict

small = 8
medium = 13
large = 16

plt.rc('font', size=large)          # controls default text sizes
plt.rc('axes', titlesize=large)     # fontsize of the axes title
plt.rc('axes', labelsize=large)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=large)    # fontsize of the tick labels
plt.rc('ytick', labelsize=large)    # fontsize of the tick labels
plt.rc('legend', fontsize=large)    # legend fontsize
plt.rc('figure', titlesize=large)  # fontsize of the figure title

# Function to calculate the learning curve
def l_curve(estim,score,estim_name,params,x,y,best_score,cv):
    lc=LearningCurveDisplay.from_estimator(estim,x,y,cv=cv,shuffle=True,random_state=42,\
    score_type="both",scoring=score,ax=plt.gca(),train_sizes=np.linspace(.1,1,9),\
    line_kw= {"marker": "o"})
    plt.savefig('test.png')
    plt.clf()

    _, axes = plt.subplots(1, 3, figsize=(30, 8))
# Add best_params_ and best_score_ to the figure
    params['SCORE']=round(best_score,3)
    _.suptitle(params)
    axes[0].set_title('Learning curve '+estim_name)
    axes[0].set_xlabel("Training samples")
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
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel("Fit times")
    axes[1].set_title("Scalability of the model")
    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel(str(score))
    axes[2].set_title("Performance of the model")
    if estim_name == 'TweedieRegressor':
        plt.savefig(estim_name+'_power'+str(estim.get_params()['regressor__model__power'])+'_'+str(score)+'_lc.png')
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
        short_score,classification,class_dim,cv,unique,scal):
# By default, no custom score and no reference scores are defined
    custom_score=False
    ref_score=False
    print('\n  Set with',x.shape[0],'samples and',x.shape[1],'features')
    print('  Training set size: ',x_train.shape[0])
    print('  Test set size: ',x_test.shape[0],'\n')
    print('  ~~~ Tuning of the parameters ~~~')
    print('   Scores performed on test set')
# Set the parameters by cross-validation
    if classification:
        if class_dim == 2:
            custom_score=make_scorer(precision_score,average='weighted',labels=['Single_layered'])
            train_scores=[custom_score,'accuracy','balanced_accuracy','f1_weighted',\
                    'precision_weighted','recall_weighted','roc_auc']
            ref_score='f1_weighted'
#            ref_score=custom_score
        else:
            custom_score=make_scorer(precision_score,average='weighted',labels=['Ideal'])
            train_scores=[custom_score,'balanced_accuracy','accuracy','f1_weighted',\
             'precision_weighted','recall_weighted']
#            train_scores=['balanced_accuracy','accuracy','f1_weighted',\
#             'precision_weighted','recall_weighted']
# Declare scoring used as reference score
            ref_score=custom_score
#            ref_score='f1_weighted'
# Scores for regression fit
    else:
# If short_score=True, evaluate only scores. If False, also neg
        train_scores=['explained_variance','max_error','r2',\
                'neg_mean_absolute_error','neg_mean_squared_error',\
                'neg_mean_absolute_percentage_error']
        neg=['neg_mean_absolute_error','neg_mean_squared_error',\
        'neg_root_mean_squared_error','neg_mean_squared_log_error',\
        'neg_median_absolute_error','neg_mean_gamma_deviance','neg_mean_absolute_percentage_error']
# Define ref_score in case of Tree estimator, to avoid looping over all scores
# (very expensive)
        ref_score='r2'
# Evaluate all scores
        if not short_score:
            train_scores=train_scores+neg
    estim_name=estimator.__class__.__name__
    if estim_name=='SVR':
        tuned_parameters = [{'regressor__model__kernel': ['rbf','linear','sigmoid'],
            'regressor__model__C': [.001,.01,.1,1, 10, 100, 1000]},
            {'regressor__model__kernel': ['poly'],
            'regressor__model__C':[.001,.01,.1,1, 10, 100],
            'regressor__model__degree': [1,2,3,4,5]}]
    elif estim_name=='SVC':
        tuned_parameters = [{'model__kernel': ['rbf','linear','sigmoid'],
            'model__C': [.001,.01,.1,1, 10, 100, 1000]},
            {'model__kernel': ['poly'],
            'model__C':[.001,.01,.1,1, 10, 100],
            'model__degree': [1,2,3,4,5]}]
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
        tuned_parameters = [{'model__penalty':['l2'],
            'model__solver':['liblinear','newton-cholesky','newton-cg','lbfgs','sag','saga'],
            'model__class_weight':[None,'balanced'],'model__C':np.geomspace(1e-6,100.0,9)},
        {'model__penalty':['l1'],
            'model__solver':['liblinear','saga'], 'model__C':np.geomspace(1e-6,100.0,9),\
            'model__class_weight':[None,'balanced']},
        {'model__penalty':['elasticnet'],
            'model__solver':['saga'], 'model__C':np.geomspace(1e-6,100.0,9),\
            'model__class_weight':[None,'balanced'], 'model__l1_ratio':[.1,.3,.5,.7,.9]}]
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
        tuned_parameters = [{'regressor__model__power':[0,1,2,3],\
          'regressor__model__alpha': [.001,.005,.01,.05,0.3,.5,.7,.9,1,5,10]}]
    elif estim_name=='GaussianNB':
        tuned_parameters = [{'var_smoothing':np.geomspace(1e-10,1e-2,9)}]
    elif estim_name=='BernoulliNB':
        tuned_parameters = [{'alpha': [1e-5, 1e-4, 1e-3, 1e-2,.1,1]}]
    elif estim_name=='KNeighborsRegressor':
        tuned_parameters = [{'regressor__model__n_neighbors':[4,5,6,8,10,12,14,16,18],\
        'regressor__model__metric':['cityblock','cosine','minkowsky','euclidean','haversine',\
        'l1','l2','manhattan','nan_euclidean']}]
    elif estim_name=='KNeighborsClassifier':
        tuned_parameters = [{'model__n_neighbors':[4,5,6,8,10,12,14,16,18],\
        'model__metric':['cityblock','cosine','minkowsky','euclidean','haversine',\
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
            tuned_parameters = {'max_depth':[3,5,7,10],
          'max_leaf_nodes':[3,5,10,15],
          'min_samples_leaf':[1,3,5,10,15],
          'min_samples_split':[8,10,12,18],
          'min_weight_fraction_leaf':[0,0.3,0.6,0.9],
          'min_impurity_decrease':[0,0.3,0.6,0.9],
          'ccp_alpha':np.linspace(0,0.04,8)}
##        'criterion': ['squared_error','friedman_mse','absolute_error','poisson']}
# ETC is incomplete
        elif estim_name=='ExtraTreesClassifier':
            tuned_parameters = {'max_depth':[None,1,2,3,4,5],'criterion':['gini','entropy'],
            'min_samples_leaf':[3,5,10,15,20],
            'min_samples_split':[8,10,12,18,20,16],
            'min_weight_fraction_leaf':[0.0,0.2,0.4,0.6,0.8],
            'max_features':['sqrt','log2',None]}
        elif estim_name=='DecisionTreeClassifier':
# Ideal: loose stopping criteria, good prunning to prevent overfitting.
# More prunning = more general, less accurate
# Stopping criteria: max_depth, min_samples_leaf, min_samples_split 
# Prunning methods: min_weight_fraction_leaf, min_impurity_decrease
            tuned_parameters = {'max_depth':[3,5,7,10],
          'max_leaf_nodes':[3,5,10,15],
          'min_samples_leaf':[1,3,5,10,15],
          'min_samples_split':[8,10,12,18],
          'min_weight_fraction_leaf':[0,0.3,0.6,0.9],
          'min_impurity_decrease':[0,0.3,0.6,0.9],
          'class_weight':[None,'balanced'],
          'ccp_alpha':np.linspace(0,0.04,8)}
#          'criterion':['gini','entropy','log_loss']}
    elif estim_name=='MLPRegressor':
        tuned_parameters = [{'regressor__model__activation':['identity','logistic','tanh','relu'],
                    'regressor__model__learning_rate':['constant','invscaling','adaptive'],
            'regressor__model__alpha': [0,.001,.005,.01,.05,.1,.5,1,5,10]}]
    elif estim_name=='MLPClassifier':
        tuned_parameters = [{'model__activation':['identity','logistic','tanh','relu'],
                    'model__learning_rate':['constant','invscaling','adaptive'],
            'model__alpha': [0,.001,.005,.01,.05,.1,.5,1,5,10]}]

# Perform cross_validate for all scores    
    scores=train_scores
# If a reference score has been defined, perform GridSearchCV only for this score
    if ref_score:
        train_scores=[ref_score]
# Loop over all scorers defined for training
    for score in train_scores:
# Define a pipeline for scaling of input variables        
        pipeline = Pipeline([('scale',scal),('model', estimator)])
        if not classification:
# Transform target variable too with the same transformer as the input variables
# (only for regression tasks)
            reg_estimator=TransformedTargetRegressor(regressor=pipeline,transformer=scal)
        else:
# In case of classification, perform scaling but not on the target
            reg_estimator=pipeline
# Perform grid search for the best hyperparameter set with CV
        clf = GridSearchCV(reg_estimator, tuned_parameters,
                scoring=score,cv=cv,return_train_score=True)
# Fit the model with the best hyperparameter set
        clf.fit(x_train, y_train)
# Prepare the dataframe with results, to print the mean and standard deviation
# of the test score for the best set of parameters
        results_df = pd.DataFrame(clf.cv_results_)
        results_df = results_df.sort_values(by=['rank_test_score'])
        results_df = (results_df.set_index(results_df["params"].apply(
            lambda v: "_".join(str(val) for val in v.values()))))
# Print results
        if not classification and score in neg:
                print('\n ·',score,':',-round(clf.score(x_test,y_test),4))
        else:
            print('\n ·',score,':',round(clf.score(x_test,y_test),4))
        print(results_df[['mean_test_score','std_test_score','mean_train_score','std_train_score']].head(1))
        print(clf.best_estimator_)
# Save best estimator with parameters tuned according to ref_score
        if score==ref_score:
            clf_best=clf.best_estimator_
# Validation curve only if regularization parameter is defined for estimator
        key=None
        if estim_name == 'MLPClassifier' or estim_name == 'TweedieRegressor'\
        or estim_name == 'MLPRegressor' or estim_name == 'BernoulliNB':
            key='alpha'
        elif estim_name == 'SVC' or estim_name=='LogisticRegressor':
            key='C'
##        if key:
##            if score == 'neg_mean_squared_error' or score == 'r2'\
##            or score == 'f1_weighted':
##                val_curve(clf.best_estimator_,score,estim_name,key,x,y,cv)

# Function to print the score with respect to the fold: informs whether the
# results are dependent on the fold
##        covar_print(results_df,score,estim_name)
##        bayesian_test(results_df,x,y)

# Uncomment to print all grid results
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
    print('\n            Summary           ')
    print('           ---------          ')
    if class_dim == 2:
        print('\n  ~~~ Binary classification problem ~~~')
        y_predict=clf_best.predict(x_test)
        print(clf.best_estimator_)
        y_test=y_test.to_numpy()
#        print('Accuracy', metrics.accuracy_score(y_test, y_predict))
#        print('Precision', metrics.precision_score(y_test, y_predict))
#        print('Recall', metrics.recall_score(y_test, y_predict))
#        print('ROC_AUC', metrics.roc_auc_score(y_test, y_predict))
        cm=confusion_matrix(y_test, y_predict)
        plot.plot_confmat_multi(y_test,y_predict,estim_name,unique)
        plot.plot_roc(y_test,clf_best.predict_proba(x_test)[:,1],unique,estimator)
        tn,fp,fn,tp=cm.ravel()
        print('  Confusion matrix plotted to cf_',str(estim_name),'.png')
        print('    True negatives',tn)
        print('    False positives',fp)
        print('    False negatives',fn)
        print('    True positives',tp)
# If binary classification problem, plot confusion matrix
#        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
#        disp.plot()
#        plt.savefig(estim_name+'_cm.png')
# If there is a custom score, convert it to dictionary to be able to perform
# cross_validate
        if custom_score:
            scores[0]={'custom_score':custom_score}
        print()
    elif class_dim > 2:
        if estim_name=='LogisticRegression':
# Calculate importance of the features from .coef_ attribute
            importance=clf_best.named_steps['model'].coef_[0]
# Print and plot importances in bar chart
            plot.importance_bar(feat_names,importance)
        clf_best.fit(x_train, y_train)
        print('\n  ~~~ Multi-class classification problem ~~~')
        print('\n   - Using',ref_score,'as reference score metric\n')
# Perform predict on the test set
        y_predict=clf_best.predict(x_test)
# Print the actual set of parameters after CV tuning
# If there is a custom score, convert it to dictionary to be able to perform
# cross_validate
        if custom_score:
            scores[0]={'custom_score':custom_score}
# Plot confusion matrix for multilabel (labels passed in unique variable)
        plot.plot_confmat_multi(y_test,y_predict,estim_name,unique)
        print('   - Confusion matrix plotted to cf_',estim_name+'.png')
# In case it is a regression problem
    elif not classification:
# Learning curve with optimal hyperparameters and for loss/r2 scoring functions
        score_lc=['neg_mean_squared_error','r2']
        for i in score_lc:
            l_curve(clf_best,i,estim_name,clf.best_params_,x,y,clf.best_score_,cv)
# Perform CV with all the scoring functions, in order to remove random effects
# on the evaluation of the accuracy of the model
    predict_evaluate.cv_perform(clf_best,x,y,cv,scores,classification,class_dim)
# Interpret model with LIME
    predict_evaluate.interpret(clf_best,x_train,y_train,x_test,y_test,classification,feat_names,estim_name)
# Save model
    model_filename = estim_name+'_model.joblib'
    joblib.dump(clf_best, model_filename)
    print('\n Model saved to',estim_name+'_model.joblib')
# Perform predict on the test set
##    y_predict=clf_best.predict(x_test)
##    y_test=y_test.to_numpy()
##    print('true    pred')
##    for i in range(len(y_test)):
##        print(y_test[i],y_predict[i])
##    print('     r2', metrics.r2_score(y_test, y_predict))
##    print('     Max error', metrics.max_error(y_test, y_predict))
##        print('     MAE', metrics.mean_absolute_error(y_test, y_predict))
##        print('     MSE', metrics.mean_squared_error(y_test, y_predict))
##        print('     MA%E', metrics.mean_absolute_percentage_error(y_test, y_predict))
    return clf.best_params_

def trainmod(x,y,feat_names,short_score,classification,estimator,cv,scal):
    estimators_list= []
    class_dim=0
    for i in estimator:
        if classification:
# If it is a classification problem, find out number of classes and counts
            unique, counts=np.unique(y,return_counts=True)
            print('          ',len(unique),'clases')
            print('        {:10s} {:6s}'.format('Class','Counts'))
            print('       -------------------')
# Store number of classes in class_dim
            class_dim= len(unique)
            for j in range(class_dim):
                print('      {:12s} {:4d}'.format(unique[j],counts[j]))
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
                estimators_list.append(DTC(random_state=42))
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
                estimators_list.append(MLPC(random_state=42,max_iter=10000))
        else:
            unique=False
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
                estimators_list.append(DTR(random_state=42))
            elif i == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor as GPR
                estimators_list.append(GPR(n_restarts_optimizer=0))
            elif i == 'linr':
                from sklearn.linear_model import TweedieRegressor as TR
                estimators_list.append(TR(power=0))
            elif i == 'nn':
                from sklearn.neural_network import MLPRegressor as MLPR
                estimators_list.append(MLPR(random_state=42,solver='lbfgs'))
# Create train and test sets from the original dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.2, random_state=42)
# Perform scaling only on x (y is transformed later for regression tasks)
#    if scal:
#        x_train = scal.fit_transform(x_train)
#        x_test = scal.transform(x_test)
#        x_train = scal.fit_transform(x_train)
#        x_test = scal.transform(x_test)

    print('\n -> Tuning hyperparameters:')
    print('   All reported scores performed on the test set\n')
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
        print('   Score:',i.score(x_test,y_test))
# Perform cross-validation
        pars=gridsearchcv(i,x_train,y_train,x_test,y_test,x,y,feat_names,\
        short_score,classification,class_dim,cv,unique,scal)
