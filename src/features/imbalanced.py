from sklearn.model_selection import cross_validate, cross_val_predict

import src.data.Dataset as dt

import matplotlib.pyplot as plt
import pandas as pd

from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss

from src.features.preprocess_features import PreprocessFeatures


class ImbalancedStrategy:
    def __init__(self, st):
        self.strategy = []
        if(st == 'SMOTE'):
            self.strategy = SMOTE('minority')
        elif(st == 'over'):
            self.strategy = RandomOverSampler(random_state=0)
        elif(st == 'under'):
            self.strategy = NearMiss()

    def get_imbalanced_strategy(self):
        return ('imbalanced_strategy', self.strategy)

if __name__ == '__main__':
    base = dt.Dataset('cook')

    prepro = PreprocessFeatures(base)
    imb = ImbalancedStrategy('over')

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

        rf = Pipeline(steps=[('preprocessor', prepro.get_preprocessor()), imb.get_imbalanced_strategy(), ('classifier', classifier)])
        out = cross_val_predict(rf, base.features, base.target, cv = base.cv , verbose = 100, method='predict_proba')

        prob = pd.DataFrame(out, columns=[1, 2, 3, 4, 5, 6, 7, 8])

        y_pred = pd.Series(prob.eq(prob.max(1), axis=0).dot(prob.columns)).astype(float)

        #
        # print("============" + str(classifier) + "==================")
        # for f in range(0, folds.__len__(), 2):
        #     X_train = folds[f].drop(columns=['target'])
        #     y_train = list(folds[f]['target'].apply(int).values)
        #
        #     X_test = folds[f + 1].drop(columns=['target'])
        #     y_test = list(folds[f + 1]['target'].apply(int).values)
        #
        #     imb = imb.imbalanced_strategy('over')
        #     X_train, y_train = imb.fit_sample(X_train, y_train)
        #     X_train = pd.DataFrame(X_train)
        #     y_train = list(y_train)
        #
        #     rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        #
        #     rf.fit(X_train, y_train)
        #
        #     prob = pd.DataFrame(rf.predict_proba(X_test), columns=[1, 2, 3, 4, 5, 6, 7, 8])
        #     y_pred = pd.DataFrame(rf.predict(X_test))
        #
        #     print('recall:', recall_score(y_test, y_pred, average='weighted'), 'precision:',
        #           precision_score(y_test, y_pred, average='weighted'), ' accuracy:'
        #           , accuracy_score(y_test, y_pred))
        #
        #     print(confusion_matrix(y_test, y_pred))
        #
        #     # print(confusion_matrix(y_test, y_pred))
        #
        #     fig = plt.figure()
        #     fig.subplots_adjust(hspace=0.4, wspace=0.7)
        #
        #     y_train = pd.DataFrame(y_train)
        #     ax = fig.add_subplot(1, 3, 1)
        #     ax.set_title('Distributin train - fold ' + str(f / 2), fontsize = 6)
        #     ax.tick_params(labelsize=5)
        #     ax.bar(y_train[0].value_counts().sort_index().index, y_train[0].value_counts().sort_index().values,
        #         tick_label=y_train[0].value_counts().sort_index().index)
        #
        #     y_test = pd.DataFrame(y_test)
        #
        #     ax = fig.add_subplot(1, 3, 2)
        #     ax.set_title('Distributin test - fold ' + str(f/2), fontsize = 6)
        #     ax.tick_params(labelsize=5)
        #     ax.bar(y_test[0].value_counts().sort_index().index,y_test[0].value_counts().sort_index().values,
        #          tick_label=y_test[0].value_counts().sort_index().index)
        #
        #     y_pred = pd.DataFrame(y_pred)
        #
        #     ax = fig.add_subplot(1, 3, 3)
        #     ax.set_title('Distributin pred - fold ' + str(f / 2), fontsize = 6)
        #     ax.tick_params(labelsize=5)
        #     ax.bar(y_pred[0].value_counts().sort_index().index, y_pred[0].value_counts().sort_index().values,
        #           tick_label=y_pred[0].value_counts().sort_index().index)
        #
        #     plt.savefig('data/img/' + base + '_pred_fold_' + str(f / 2) + '.pdf')