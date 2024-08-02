Machine learning (supervised and unsupervised) with Python scikit-learn

Own-built code by Alfonso Gallo that allows for regression, binary and multiclassification ML model building, both supervised and unsupervised.

It accepts the following ML algorithms:
- Linear regression (Logistic regression for classification tasks) (LinR)
- Decission tree (DT)
- k-Nearest Neighbours (kNN)
- Support Vector Machine (SVM)
- Neural Networks (NN)
- Principal Component Analysis (PCA)
- Stochastic Gradient Descent: (SGD)
- Gaussian Naive Bayes (GNB - only for classification)
- Bernoulli Naive Bayes (BNB - only for classification)
- Gaussian process (GP - under development)

It allows for the evaluation of different metrics and preprocessing tasks:
- Strandardization and normalization approaches
- Learning curve (regression tasks only)
- Correlation circle (through MLxtend)
- Bias and variance evaluation (through MLxtend)
- Confusion matrix (classification tasks only)
- Feature selection
- Feature importance
- Feature generation

Code can be run in training mode or a model can be supplied and applied.
It is not intended for public use but rather for personal work, so therefore making use of  the code might
involve modifiying some code files.
Input is specified through a plain text file, although not all functionalities are accesible that way.

Example of a regression model training input file:
<span>#</span> classification=True                                 <span>#</span> Regression model by default  
yname='am_loading'                                                 <span>#</span> Target variable  
dropcols=['name','shear_rate','am_loading_std']                    <span>#</span> Dropping some columns from the database  
data_f='23-01-23_cathode_am_mass_reg_rem_high.csv'                 <span>#</span> Data file  
-- Model keyword description --  
Linear regression (Tweedie Regressor - regression only): linr  
<span>#</span> K-Nearest Neighbors: knn  
<span>#</span> Support Vector Machine: svm  
<span>#</span> Stochastic Gradient Descent: sgd  
<span>#</span> Decission Tree: dt  
<span>#</span> Neural network: nn  
<span>#</span> Logistic Regression (classification only): lr  
<span>#</span> Gaussian Naive Bayes (classification only): gnb  
<span>#</span> Bernoulli Naive Bayes (classification only): bnb  
<span>#</span> Gaussian process: gp <span>#</span> NOT READY!  
<span>#</span> plot_dat_hist=True                                  <span>#</span> Plot histogram with raw data  
estimator=['svm']<span>#</span>,'sgd','linr','knn','svm']          <span>#</span> Choose estimator(s)  
<span>#</span> pca_expl_var=.97                                    <span>#</span> Perform explained variance task  
<span>#</span> feat_sel=True                                       <span>#</span> Perform feature selection  
correlation_plot=True                                              <span>#</span> Plot correlation circle  
<span>#</span> gen_feat=3                                          <span>#</span> Generate # features  
<span>#</span> spline_knots=3                                      <span>#</span> Generation features settings  


Code in continuous development and more features are being added on demand.

For any question/suggestion, please refer to alfonso.gallo.bueno@gmail.com.
