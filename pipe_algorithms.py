from sklearn.model_selection import KFold

from feature_selection import recursive_feature_elimination
from read_folds import load_folds
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
import numpy

def pipele_classification(folds):
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
    ]

    for classifier in classifiers:
        print("============" + str(classifier) + "==================")
        for f in range(0, folds.__len__(), 2):
            print(folds[f])
            X_train = folds[f].drop(columns=['target'])
            y_train = list(folds[f]['target'].apply(int).values)

            X_test = folds[f + 1].drop(columns=['target'])
            y_test = list(folds[f + 1]['target'].apply(int).values)

            rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            print('recall:', recall_score(y_test, y_pred, average='weighted'), 'precision:',
                  precision_score(y_test, y_pred, average='weighted'), ' accuracy:'
                  , accuracy_score(y_test, y_pred))

            # print(confusion_matrix(y_test, y_pred))

            # fig = plt.figure()
            # fig.subplots_adjust(hspace=0.4, wspace=0.7)

            y_train = pd.DataFrame(y_train)
            # ax = fig.add_subplot(1, 3, 1)
            # ax.set_title('Distributin train - fold ' + str(f / 2), fontsize = 6)
            # ax.tick_params(labelsize=5)
            # ax.bar(y_train[0].value_counts().sort_index().index, y_train[0].value_counts().sort_index().values,
            #    tick_label=y_train[0].value_counts().sort_index().index)

            y_test = pd.DataFrame(y_test)

            # ax = fig.add_subplot(1, 3, 2)
            # ax.set_title('Distributin test - fold ' + str(f/2), fontsize = 6)
            # ax.tick_params(labelsize=5)
            # ax.bar(y_test[0].value_counts().sort_index().index,y_test[0].value_counts().sort_index().values,
            #     tick_label=y_test[0].value_counts().sort_index().index)

            y_pred = pd.DataFrame(y_pred)

            # ax = fig.add_subplot(1, 3, 3)
            # ax.set_title('Distributin pred - fold ' + str(f / 2), fontsize = 6)
            # ax.tick_params(labelsize=5)
            # ax.bar(y_pred[0].value_counts().sort_index().index, y_pred[0].value_counts().sort_index().values,
            #      tick_label=y_pred[0].value_counts().sort_index().index)

            # plt.savefig('data/img/' + base + '_pred_fold_' + str(f / 2) + '.pdf')

if __name__ == '__main__':
    base = 'cook'
    folds = load_folds(base)
    #pipele_classification(folds)

    #estimators = [('feat_selection', recursive_feature_elimination(folds))]

    #numeric_transformer = Pipeline([
    #    ('feat_selection', recursive_feature_elimination(folds)),
    #    ('classf', pipele_classification(folds))])
    #pipe = Pipeline(estimators)

    kf = KFold(n_splits=5)
    kf.split(folds[0])

    for train, test in kf.split(folds[0]):
        print("Train: %s \nTest:%s" % (train, test))
        print("Train: %s "% train.__len__())
        print("Test: %s " % test.__len__())