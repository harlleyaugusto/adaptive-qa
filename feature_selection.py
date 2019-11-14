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
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold

import numpy

def univariate_selection(folds):
    #Negative value error
    for f in range(0, folds.__len__(), 2):
        X_train = folds[f].drop(columns = ['target'])
        y_train = list(folds[f]['target'].apply(int).values)

        X_test = folds[f+1].drop(columns = ['target'])
        y_test = list(folds[f+1]['target'].apply(int).values)

        test = SelectKBest(score_func=chi2, k=4)
        fit = test.fit(X_train, y_train)

        # summarize scores
        numpy.set_printoptions(precision=3)
        print(fit.scores_)
        features = fit.transform(X_train)

        # summarize selected features
        print(features[0:5, :])

def recursive_feature_elimination(folds):
    for f in range(0, folds.__len__(), 2):
        X_train = folds[f].drop(columns = ['target'])
        y_train = list(folds[f]['target'].apply(int).values)

        X_test = folds[f+1].drop(columns = ['target'])
        y_test = list(folds[f+1]['target'].apply(int).values)

        model = GradientBoostingClassifier()
        rfe = RFE(model, 10)
        fit = rfe.fit(X_train, y_train)

        print("============== FOLD: " + str(int(f/2)) + " ===================")
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)

def variance(base):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    a = pd.DataFrame(sel.fit_transform(base))
    return a

if __name__ == '__main__':
    base_name = 'cook'
    folds = load_folds(base_name)
    base = load_base(base_name)
    base.drop(columns = ['question_id', 'target'], inplace = True)

    #univariate_selection(folds)
    #recursive_feature_elimination(folds)
    a = variance(base)
