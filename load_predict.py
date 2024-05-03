import joblib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot predicted and experimental values together
def plot_prediction(pred_y,pred_x,exp_y,exp_x,yname):
    color=['tab:blue', 'tab:orange', 'tab:green']
    for key in exp_x:
        fig, ax = plt.subplots()
        ax.scatter(exp_x[key], exp_y, c=color[2], label='experimental',
               alpha=0.3, edgecolors='none')
        ax.scatter(pred_x[key], pred_y, c=color[1], label='predicted',
               alpha=0.3, edgecolors='none')
        ax.set_xlabel(key)
        ax.set_ylabel(yname)
#        ax.set_xlim([0,20])
        plt.legend()
        plt.savefig(yname+'VS'+key+'.png')

# Function to load the previouly fitted model for production
def load(model_filename):
    print('\n -> Loading model',model_filename)
    model = joblib.load(model_filename)
    return model
#    result = loaded_model.score(X_test, Y_test)

# Read data settings for prediction
def gen_data(x,y,gen_data,yname,feat_names,n_points,model,target_property):
    estim_name=model['model'].__class__.__name__
    gen_data=pd.DataFrame(gen_data)
    pred_data=pd.DataFrame()
# pred_x contains randomly generated plotting variables (x)
    pred_x={}
# exp_y contains experimental (given) truth (y)
    exp_y=y.tolist()
# exp_x contains experimental (given) plotting variables (x)
    exp_x={}
    print('     Plotting features min max')
# Loop through all the input data (variables and values)
    for key,value in gen_data.iteritems():
# If the value is 'plot', plot this variable 
        if value[0] == 'plot':
# Find minimum and maximum values and plot within this range 
# (using n_points as number of plotting points)
            exp_x[key]=x[key].tolist()
            print('      ',key,x[key].min(),x[key].max(),'\n')
# Array randomly generated within the defined range
            array=np.random.uniform(x[key].min(), x[key].max(), size=(n_points, 1))
# Save in dictionary the 2 variables for plotting
            pred_x[key]=array[:,0].tolist()
# Loop over the number of points to predict
    for i in range(n_points):
        for key,value in pred_x.items():
            gen_data[key]=value[i]
# Append each complete sampling point (all variables required by the model)
# and store them in pandas dataframe
        pred_data=pd.concat([gen_data,pred_data.loc[:]]).reset_index(drop=True)
# Predict all points. pred_y contains all the predicted targets
    pred_y=model.predict(pred_data.values.tolist()).tolist()
    prob_y=model.predict_proba(pred_data.values.tolist()).tolist()
    plt.close()
    z=[]
    for i in prob_y:
        z.append(i[2])

#    plot_prediction(pred_y,pred_x,exp_y,exp_x,yname)
    if target_property:
        p=list(pred_x.values())[0]
        t=list(pred_x.values())[1]

        print(list(pred_x)[0])
        fig, ax = plt.subplots()
        plot=ax.scatter(p,t,c=z,cmap='RdBu')
        ax.set_xlabel(list(pred_x)[0])
        ax.set_ylabel(list(pred_x)[1])
        fig.colorbar(plot,ax=ax,orientation='vertical',label='prob('+target_property+')')
        plt.savefig(estim_name+'_colormap.png')
