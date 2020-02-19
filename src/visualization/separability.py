from sklearn.model_selection import GridSearchCV, cross_val_predict


import matplotlib.pyplot as plt
import pandas as pd

from imblearn.pipeline import Pipeline
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
import numpy as np
import src.data.Dataset as dt
import src.util.config as c

from src.features.imbalanced import ImbalancedStrategy
from src.features.preprocess_features import PreprocessFeatures

if __name__ == '__main__':
    base = dt.Dataset('cook')
    prepro = PreprocessFeatures(base)
    imb = ImbalancedStrategy('over')

    parameters = {
        "loss": ["deviance"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "min_samples_split": np.linspace(0.1, 0.5, 12),
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth": [3, 5, 8],
        "max_features": ["log2", "sqrt"],
        "criterion": ["friedman_mse", "mae"],
        "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        "n_estimators": [10]
    }

    config = c.Config()

    classifiers = [
        # KNeighborsClassifier(3),
        #SVC(kernel="rbf", C=0.025, probability=True),
        # NuSVC(probability=True),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        # AdaBoostClassifier(),
        # RandomForestClassifier(),
        #GridSearchCV(GradientBoostingClassifier(), parameters)
        #GradientBoostingClassifier()
        GaussianNB()
    ]

    prob = []

    for classifier in classifiers:
        rf = Pipeline(steps=[('preprocessor', prepro.get_preprocessor()), imb.get_imbalanced_strategy(),
                             ('classifier', classifier)])
        out = cross_val_predict(rf, base.features, base.target, cv=base.cv, verbose=100, method='predict_proba')

        prob = pd.DataFrame(out, columns=[1, 2, 3, 4, 5, 6, 7, 8])

        y_pred = pd.Series(prob.eq(prob.max(1), axis=0).dot(prob.columns)).astype(float)

        prob.plot.kde(figsize=(7, 7))

        plt.savefig(config.report + config.img_report + base.name + '_' + type(classifier).__name__ + '_density_fold_.png')
    #
    #     folds = base.get_separeted_folds()
    #
    #     print("============" + str(classifier) + "==================")
    #     for f in range(0, folds.__len__(), 2):
    #         X_train = folds[f].drop(columns=['target'])
    #         y_train = list(folds[f]['target'].apply(int).values)
    #
    #         X_test = folds[f + 1].drop(columns=['target'])
    #         y_test = list(folds[f + 1]['target'].apply(int).values)
    #
    #         rf = Pipeline(steps=[('preprocessor', prepro.get_preprocessor()), imb.get_imbalanced_strategy(),
    #                              ('classifier', classifier)])
    #
    #         rf.fit(X_train, y_train)
    #
    #         if f == 0:
    #             prob = pd.DataFrame(rf.predict_proba(X_test), columns=[1, 2, 3, 4, 5, 6, 7, 8])
    #             y_pred = pd.DataFrame(rf.predict(X_test))
    #         else:
    #             prob = prob.append(pd.DataFrame(rf.predict_proba(X_test), columns=[1, 2, 3, 4, 5, 6, 7, 8]))
    #
    #
    #     prob.plot.kde(figsize=(7,7))
    #
    #     plt.savefig(
    #         config.report + config.img_report + base.name + '_' + type(classifier).__name__ + '_density_fold_.png')

            # print('recall:', recall_score(y_test, y_pred, average='weighted'), 'precision:',
            #       precision_score(y_test, y_pred, average='weighted'), ' accuracy:'
            #       , accuracy_score(y_test, y_pred))
            #
            # print(confusion_matrix(y_test, y_pred))