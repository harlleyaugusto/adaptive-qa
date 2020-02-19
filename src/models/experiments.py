from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import src.data.Dataset as dt


if __name__ == '__main__':
    base = dt.Dataset('cook')
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # TODO: categorical and numerical issue must to be solved!

    numeric_features = base.features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = base.features.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    rf = Pipeline(steps=[('preprocessor', preprocessor), ('imbalanced', SMOTE('minority')),
                         ('classifier', GradientBoostingClassifier())])
    scores = cross_validate(rf, base.features, base.target['target'], cv=base.cv, verbose=100,
                            scoring=['recall_weighted'])

    ''' SMOTE, over and under sampling  '''
    rf = Pipeline(steps=[('preprocessor', preprocessor), ('imbalanced', SMOTE('minority')), ('classifier', GradientBoostingClassifier())])
    scores = cross_validate(rf, base.features, base.target['target'], cv=base.cv, verbose=100, scoring= ['recall_weighted'])


    rf = Pipeline(steps=[('preprocessor', preprocessor), ('imbalanced', RandomOverSampler(random_state=0)),
                         ('classifier', GradientBoostingClassifier())])
    scores = cross_validate(rf, base.features, base.target['target'], cv=base.cv, verbose=100,
                            scoring=['recall_weighted'])

    rf = Pipeline(steps=[('preprocessor', preprocessor), ('imbalanced', NearMiss()),
                         ('classifier', GradientBoostingClassifier())])
    scores = cross_validate(rf, base.features, base.target['target'], cv=base.cv, verbose=100,
                            scoring=['recall_weighted'])


    '''Feature selection'''
    rf = Pipeline(steps=[('preprocessor', preprocessor), ('feature_selection', VarianceThreshold(threshold=(.8 * (1 - .8)))),
                         ('classifier', GradientBoostingClassifier())])
    scores = cross_validate(rf, base.features, base.target['target'], cv=base.cv, verbose=100,
                            scoring=['recall_weighted'])

    rf = Pipeline(steps=[('preprocessor', preprocessor), ('feature_selection', SelectKBest(score_func = f_regression, k = 50)),
               ('classifier', GradientBoostingClassifier())])
    scores = cross_validate(rf, base.features, base.target['target'], cv=base.cv, verbose=100,
                            scoring=['recall_weighted'])
