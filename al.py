from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

def train(x,y,i,classification):
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
            estimator=KNC(n_neighbors=4)
        elif i == 'svm':
            estimator=svm.SVC()
        elif i == 'sgd':
            from sklearn.linear_model import SGDClassifier as SGDC
            estimator=SGDC(max_iter=10000)
        elif i == 'dt':
            from sklearn.tree import DecisionTreeClassifier as DTC
            estimator=DTC()
        elif i == 'lr':
            from sklearn.linear_model import LogisticRegression as LR
            estimator=LR(max_iter=20000)
        elif i == 'gp':
            from sklearn.gaussian_process import GaussianProcessClassifier as GPC
            estimator=GPC(n_restarts_optimizer=0)
    else:
        if i == 'knn':
            from sklearn.neighbors import KNeighborsRegressor as KNR
            estimator=KNR()
        elif i == 'svm':
            estimator=svm.SVR()
        elif i == 'sgd':
            from sklearn.linear_model import SGDRegressor as SGDR
            estimator=SGDR()
        elif i == 'dt':
            from sklearn.tree import DecisionTreeRegressor as DTR
            estimator=DTR()
        elif i == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor as GPR
            estimator=GPR(n_restarts_optimizer=0)

    learner = ActiveLearner(
    estimator=estimator,
    query_strategy=uncertainty_sampling,
    X_training=x, y_training=y)
    query_id, query_sample=learner.query(x)
    print(query_id, query_sample)
#    learner.teach(x,y)
