from autosklearn.metrics import recall

from read_folds import load_folds, load_base
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import VarianceThreshold

import numpy

def univariate_selection(folds, k):
    #Negative value error
    folds_trans = []
    for f in range(0, folds.__len__(), 2):
        X_train = folds[f].drop(columns = ['target'])
        y_train = list(folds[f]['target'].apply(int).values)

        X_test = folds[f+1].drop(columns = ['target'])
        y_test = list(folds[f+1]['target'].apply(int).values)

        test = SelectKBest(score_func = f_regression, k = k)
        fit = test.fit(X_train, y_train)

        X_train = pd.DataFrame(fit.transform(X_train))
        X_test = pd.DataFrame(fit.transform(X_test))

        X_train['target'] = y_train
        X_test['target'] = y_test

        folds_trans.append(X_train)
        folds_trans.append(X_test)

    return folds_trans

def recursive_feature_elimination(folds):

    model = GradientBoostingClassifier()
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring="precision_weighted")

    # Using just the first folds

    X_train = folds[0].drop(columns=['target']).head(10)
    y_train = list(folds[0]['target'].head(10).apply(int).values)

    rfecv = rfecv.fit(X_train, y_train)

    for f in range(0, folds.__len__(), 1):
        y = list(folds[f]['target'].apply(int).values)
        folds[f]=folds[f][folds[f].drop(columns=['target']).columns[rfecv.support_]]
        folds[f]['target'] = y

    return folds


def variance(base):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))


    a = pd.DataFrame(sel.fit_transform(base))
    return a

if __name__ == '__main__':
    base_name = 'cook'
    folds = load_folds(base_name)
    #base = load_base(base_name)
    #base.drop(columns = ['question_id', 'target'], inplace = True)

    #folds = univariate_selection(folds, 50)
    folds = variance(folds)
    #folds = recursive_feature_elimination(folds)
    #a = variance(base)

    #pipele_classification(folds)

