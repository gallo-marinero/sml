from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import TweedieRegressor as TR
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import train_test_split, cross_validate
from mlxtend.evaluate import bias_variance_decomp
import math

# Calculate bias and variance (as an estimation of over- and underfitting)
def interpret(model,x_train,y_train,x_test,y_test,classification,feat_names,estimator):
    if classification:
        loss='0-1_loss'
# Encode the text labels to integers        
        y_train=LabelEncoder().fit_transform(y_train)
        y_test=LabelEncoder().fit_transform(y_test)
    else:
        loss='mse'
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model,
            x_train, y_train, x_test, y_test, loss=loss, num_rounds=50, random_seed=1)
    print('\n Over/underfitting evaluaton')
    print(f"    Average expected loss: {avg_expected_loss.round(3)}")
    print(f"    Average bias: {avg_bias.round(3)}")
    print(f"    Average variance: {avg_var.round(3)}")

def pred_eval(x,y,model):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
# Fit the data to a Linear regression
    model.fit(x_train,y_train)
# Evaluate the model
    eval_model=model.predict(x_test)
# Evaluate the prediction
#    mae=mean_absolute_error(y_test, eval_model)
    return x_train,y_train,x_test, y_test, eval_model

# Function for performing CV on the metrics for evaluation of model performance 
def cv_perform(estimator,x,y,cv,scores,classification,class_dim):
    custom=False
# Loop over scores to check if there is custom score. If so, remove it,
# otherwise cross_validate throws error
    for i in scores:
        if type(i) is dict:
            custom=i
            scores.remove(i)
    cv_score=cross_validate(estimator,x,y,cv=cv,scoring=scores,return_train_score=True)
    print('\n Estimator: ',estimator)
    print('\n Evaluating accuracy with CV')
    print('    {:20s} {:6s} {:6s} {:6s} {:6s}'.format('Score','Train','Std','Test','Std'))
    print('    -------------------------------------------------')
# Print actual set of parameters after CV tuning            
    if not classification:
        for i in scores:
            test_key='test_'+i
            train_key='train_'+i
            if 'neg_mean_squared_error' in i:
                print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format('RMSE',
            round(math.sqrt(abs(cv_score[train_key].mean())),3),\
            round(math.sqrt(abs(cv_score[train_key].std())),3),\
            round(math.sqrt(abs(cv_score[test_key].mean())),3),\
            round(math.sqrt(abs(cv_score[test_key].std())),3)))
            elif 'neg_mean_absolute_error' in i:
                print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format('MAE',
            round(abs(cv_score[train_key].mean()),3),round(cv_score[train_key].std(),3),
            round(abs(cv_score[test_key].mean()),3),round(cv_score[test_key].std(),3)))
            elif 'neg_mean_absolute_percentage_error' in i:
                print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format('MA%E',
            round(abs(cv_score[train_key].mean())*100,3),round(cv_score[train_key].std()*100,3),
            round(abs(cv_score[test_key].mean())*100,3),round(cv_score[test_key].std()*100,3)))
            elif 'max_error' in i:
                print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format('Max error',
            round(abs(cv_score[train_key].mean()),3),round(cv_score[train_key].std(),3),
            round(abs(cv_score[test_key].mean()),3),round(cv_score[test_key].std(),3)))
            elif 'explained_variance' in i:
                print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format('Exp var',
            round(abs(cv_score[train_key].mean()),3),round(cv_score[train_key].std(),3),
            round(abs(cv_score[test_key].mean()),3),round(cv_score[test_key].std(),3)))
            else:
                print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format(i,
            round(cv_score[train_key].mean(),3),round(cv_score[train_key].std(),3),
            round(cv_score[test_key].mean(),3),round(cv_score[test_key].std(),3)))
#    elif class_dim > 2:
    else:
        for i in scores:
            test_key='test_'+i
            train_key='train_'+i
            print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format(i,
            round(cv_score[train_key].mean(),3),round(cv_score[train_key].std(),3),
            round(cv_score[test_key].mean(),3),round(cv_score[test_key].std(),3)))
# Create special instance for custom scores (only for multiclass by now
        if custom:
            cv_score=cross_validate(estimator,x,y,cv=cv,scoring=custom,return_train_score=True)
            test_key='test_custom_score'
            train_key='train_custom_score'
            print('    {:20s} {:6.3f}{:6.3f} {:6.3f}{:6.3f}'.format('custom_score',
            round(cv_score[train_key].mean(),3),round(cv_score[train_key].std(),3),
            round(cv_score[test_key].mean(),3),round(cv_score[test_key].std(),3)))
