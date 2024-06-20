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
# classification=True                                              # Regression model by default
yname='am_loading'                                                 # Target variable
dropcols=['name','shear_rate','am_loading_std']                    # Dropping some columns from the database
data_f='23-01-23_cathode_am_mass_reg_rem_high.csv'                 # Data file
# -- Model keyword description --
# Linear regression (Tweedie Regressor - regression only): linr
# K-Nearest Neighbors: knn
# Support Vector Machine: svm
# Stochastic Gradient Descent: sgd
# Decission Tree: dt
# Neural network: nn
# Logistic Regression (classification only): lr
# Gaussian Naive Bayes (classification only): gnb
# Bernoulli Naive Bayes (classification only): bnb
# Gaussian process: gp # NOT READY!
# plot_dat_hist=True                                                # Plot histogram with raw data
estimator=['svm']#,'sgd','linr','knn','svm']                        # Choose estimator(s)
# pca_expl_var=.97                                                  # Perform explained variance task
# feat_sel=True                                                     # Perform feature selection
correlation_plot=True                                               # Plot correlation circle
# gen_feat=3                                                        # Generate # features
# spline_knots=3                                                    # Generation features settings


Code in continuous development and more features are being added on demand.

For any question/suggestion, please refer to alfonso.gallo.bueno@gmail.com.
