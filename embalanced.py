from class_to_question import load_folds
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
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import class_weight
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss

if __name__ == '__main__':
    base = 'cook'
    folds = load_folds(base, [0])

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # TODO: categorical and numerical issue must to be solved!

    numeric_features = folds[0].drop(columns=['target']).select_dtypes(include=['int64', 'float64']).columns
    categorical_features = folds[0].drop(columns=['target']).select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="rbf", C=0.025, probability=True),
        # NuSVC(probability=True),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        # AdaBoostClassifier(),
        # RandomForestClassifier(),
        GradientBoostingClassifier()
        #GaussianNB()
    ]

    prob = []

    for classifier in classifiers:
        print("============" + str(classifier) + "==================")
        for f in range(0, folds.__len__(), 2):
            X_train = folds[f].drop(columns=['target'])
            y_train = list(folds[f]['target'].apply(int).values)

            X_test = folds[f + 1].drop(columns=['target'])
            y_test = list(folds[f + 1]['target'].apply(int).values)

            #imb = SMOTE('minority')
            #imb = RandomOverSampler(random_state=0)
            imb = NearMiss()
            X_train, y_train = imb.fit_sample(X_train, y_train)
            X_train = pd.DataFrame(X_train)
            y_train = list(y_train)

            rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

            #sample_weight = class_weight.compute_sample_weight('balanced', y_train)

            #rf.fit(X_train, y_train, **{'classifier__sample_weight' : sample_weight})
            rf.fit(X_train, y_train)

            prob = pd.DataFrame(rf.predict_proba(X_test), columns=[1, 2, 3, 4, 5, 6, 7, 8])
            y_pred = pd.DataFrame(rf.predict(X_test))

            prob.plot.density()

            plt.savefig('data/img/' + base +'_density_fold_'+ str(int(f / 2)) + '.pdf')

            print('recall:', recall_score(y_test, y_pred, average='weighted'), 'precision:',
                  precision_score(y_test, y_pred, average='weighted'), ' accuracy:'
                  , accuracy_score(y_test, y_pred))

            print(confusion_matrix(y_test, y_pred))

    #prob = pd.concat(prob).reset_index();