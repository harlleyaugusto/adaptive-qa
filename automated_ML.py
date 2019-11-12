
from class_to_question import load_folds
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy

import autosklearn.classification

if __name__ == '__main__':
    base = 'cook'
    folds = load_folds(base)

    '''
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
        output_folder='/tmp/autosklearn_parallel_1_example_out',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
        n_jobs=4,
        seed=5,
        delete_output_folder_after_terminate=False,
        delete_tmp_folder_after_terminate=False,
    )
    '''

    automl = autosklearn.classification.AutoSklearnClassifier()
    X_train = folds[4].drop(columns=['target'])
    y_train = list(folds[4]['target'].apply(int).values)

    X_test = folds[5].drop(columns=['target'])
    y_test = list(folds[5]['target'].apply(int).values)

    automl.fit(X_train, y_train, dataset_name='cook')

    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", accuracy_score(y_test, predictions))
