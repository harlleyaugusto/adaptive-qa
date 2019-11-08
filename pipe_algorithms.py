from class_to_question import load_folds
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    folds = load_folds("cook")

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    #TODO: categorical and numerical issue must to be solved!

    numeric_features = folds[0].drop(columns = ['target']).select_dtypes(include=['int64', 'float64']).columns
    categorical_features = folds[0].drop(columns = ['target']).select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    for f in range(0, folds.__len__(), 2):
        X_train = folds[f].drop(columns = ['target'])
        y_train = list(folds[f]['target'].apply(int).values)

        X_test = folds[f+1].drop(columns = ['target'])
        y_test = list(folds[f+1]['target'].apply(int).values)

        rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        print("model score: %.3f" % rf.score(X_test, y_test))
        #rf.score(X_test, y_test)
