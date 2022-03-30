from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import TweedieRegressor as TR
from sklearn.model_selection import train_test_split

def pred_eval(x,y,model):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,
            random_state=42)
# Fit the data to a Linear regression
    model.fit(x_train,y_train)
# Evaluate the model
    eval_model=model.predict(x_test)
# Evaluate the prediction
#    mae=mean_absolute_error(y_test, eval_model)
    return x_train,y_train,x_test, y_test, eval_model

