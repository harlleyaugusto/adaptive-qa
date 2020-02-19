import src.data.Dataset as dt

import pandas as pd

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_validate

from src.features.preprocess_features import PreprocessFeatures

class FeatureSelection:
    def __init__(self):
        self.sel = []
        self.univariate_selection = []
        self.recursive_feature_elimination = []

    # remove all features that are either one or zero (on or off) in more than variance% of the samples
    def set_sel(self, variance):
        v = variance if variance <= 1.0 or variance > 0 else 0.8
        self.sel = VarianceThreshold(threshold=(v * (1 - v)))

    def set_univariate_selectio(self, sf, k):
        self.univariate_selection = SelectKBest(score_func=sf, k=k)

    def set_recursive_feature_elimination(self, model, st, sc = "precision_weighted"):
        self.univariate_selection = RFECV(estimator = model, step= st,  scoring= sc)

    def get_methods(self):
        l = []
        if self.sel is not None:
            l.append(['variance', self.sel])
        if self.recursive_feature_elimination is not None:
            l.append(['recursive_feature_elimination', self.recursive_feature_elimination])
        if self.univariate_selection is not None:
            l.append(['univariate_selection', self.univariate_selection])

if __name__ == '__main__':
    base = dt.Dataset('cook')
    prepro = PreprocessFeatures(base)
    fs = FeatureSelection()
    fs.set_sel(0.8)
    rf = Pipeline(steps=[('preprocessor', prepro.get_preprocessor())])

    if fs.get_methods() is not None:
        rf.steps.append(fs.get_methods())

    rf.steps.append(('classifier', GradientBoostingClassifier()))


