from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import predict_evaluate

def plot_skb(x_train,y_train,x_test):
    print('Bar plots saved.')
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

def skb(x,y):
# Define the evaluation method and model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    model=LinearRegression()
    for i in range(3):
        if i==0:
            print('\n ~ Fit with all features ~ ')
            x_train,y_train,x_test,y_test,eval_model=predict_evaluate.lr(x,y)
##            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,
 ##           random_state=42)
# Fit the data to a Linear regression
  ##          model.fit(x_train,y_train)
# Evaluate the model
   ##         eval_model=model.predict(x_test)
# Evaluate the prediction
            mae=mean_absolute_error(y_test, eval_model)
            print('Number of features: ',x_train.shape[1])
            print('Mean Absolute Error: {:5.3f}'.format(mae))
# Plot the score for each feature
            plot_skb(x_train,y_train,x_test)

# Configure to select all features
        elif i !=0:
            if i == 1:
                print('\n ~ Features according to mutual information ~ ')
                fs = SelectKBest(score_func=mutual_info_regression)
            elif i == 2:
                print('\n ~ Features according to F regression ~ ')
                fs = SelectKBest(score_func=f_regression)
            pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# Define the grid: explore from 2 features to all, in steps of 2
            grid = dict()
            grid['sel__k'] = [i for i in range(2, x.shape[1]+1,2)]
# Define the grid search: use MAE as scoring function
            search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', cv=cv)
# Perform the search
            results = search.fit(x, y)
# What are scores for the features
#            for i in range(len(fs.scores_)):
#                print('{:15.13s} {:6.4f}'.format(x.columns[i], fs.scores_[i]))
# Plot the scores
#            pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
#            pyplot.show()
            # summarize best
            print('Mean Absolute Error: {:5.3f}'.format(results.best_score_))
            print('Best Config: {}'.format(results.best_params_))
# Summarize all
            means = results.cv_results_['mean_test_score']
            params = results.cv_results_['params']
            for mean, param in zip(means, params):
                print(">%.3f with: %r" % (mean, param))
