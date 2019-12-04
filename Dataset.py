import pandas as pd
import os

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_files
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from read_folds import load_folds

class Dataset:
    path = 'data/questionExtraction/'

    def __init__(self, name):
        self.base_name = name
        self.features = []
        self.target = []
        #self.folds = []
        self.cv = []
        self.load_base()


    def load_base(self, fold_num=None):
        base_path = self.path + self.base_name + "/svm/class"

        train_indices = []
        test_indices = []

        for f in (range(0, 5) if fold_num is None or fold_num.__len__() == 0 else fold_num):
            files = [base_path + '/class_fold_' + str(f) + '_train_all_.svm']
            files.append(base_path + '/class_fold_' + str(f) + '_test_all_.svm')
            matrix_folds = load_svmlight_files(files, query_id=True)

            # Get train and target
            a = pd.DataFrame(matrix_folds[0].toarray())
            a['target'] = matrix_folds[1]
            a['id_query'] = matrix_folds[2]
            a = a.reset_index()
            a = a.set_index('id_query')
            a = a.drop(columns=['index'])

            #base.features.index.get_indexer_for(base.features.loc[base.cv[0][0], :].index)

            # Get test and target
            b = pd.DataFrame(matrix_folds[3].toarray())
            b['target'] = matrix_folds[4]
            b['id_query'] = matrix_folds[5]
            b = b.reset_index()
            b = b.set_index('id_query')
            b = b.drop(columns=['index'])
            if f == 0:
                self.features = pd.concat([a, b])
                self.target = self.features[['target']]
                self.features = self.features.drop(columns = ['target'])

            train_indices.append(self.features.index.get_indexer_for(a.index))
            test_indices.append(self.features.index.get_indexer_for(b.index))


        self.cv = (list(zip(train_indices, test_indices)))

    def getFolds(self):
        folds = []
        for f in range(0, self.cv.__len__()):
            for tt in range(0, 2):
                folds.append(self.features.iloc[base.cv[f][tt]].join(self.target, how = 'inner'))

        return folds

if __name__ == '__main__':
    base = Dataset('cook')
    #folds = load_folds('cook')

    #clf = svm.SVC(kernel='linear', C=1, random_state=0)
    #scores = cross_validate(clf, base.features, base.target['target'], scoring='f1_macro', cv =2, verbose = True)


    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # TODO: categorical and numerical issue must to be solved!

    numeric_features = base.features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = base.features.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    rf = Pipeline(steps=[('preprocessor', preprocessor), ('imbalanced', SMOTE('minority')), ('classifier', GradientBoostingClassifier())])
    scores = cross_validate(rf, base.features, base.target['target'], cv=base.cv, verbose=100)