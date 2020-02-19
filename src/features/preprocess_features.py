from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.data.Dataset as dt
from imblearn.pipeline import Pipeline

class PreprocessFeatures:

    def __init__(self, base):
        self.numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

        self.categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.numeric_features = base.features.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_features = base.features.select_dtypes(include=['object']).columns


    def get_preprocessor(self):
        return ColumnTransformer(transformers=[('num', self.numeric_transformer, self.numeric_features),
                                                   ('cat', self.categorical_transformer, self.categorical_features)])

if __name__ == '__main__':

    #Just testing
    base = dt.Dataset('cook')
    prepro = PreprocessFeatures(base)

    rf = Pipeline(steps=[('preprocessor', prepro.get_preprocessor()), ('classifier', GradientBoostingClassifier())])

    scores = cross_validate(rf, base.features, base.target, cv=base.cv, verbose=100,
                            scoring=['recall_weighted'])

