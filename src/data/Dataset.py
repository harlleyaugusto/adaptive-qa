import pandas as pd

from sklearn.datasets import load_svmlight_files
import src.util.config as c


class Dataset:
    config = c.Config()

    def __init__(self, name, inter = False):
        self.name = name

        # *all* folds (train and test) into features
        self.features = []

        # *all* targets (train and test) into target
        self.target = []

        # cv stores the index of each fold (train and test)
        self.cv = []

        self.load_base()


    def load_base(self):

        base_path = self.config.data_dir + self.config.processed_data + self.name

        train_indices = []
        test_indices = []

        for f in (range(0, 5)):
            files = [base_path + '/class_fold_' + str(f) + '_train_all_.svm']
            files.append(base_path + '/class_fold_' + str(f) + '_test_all_.svm')
            matrix_folds = load_svmlight_files(files, query_id=True)

            # Get train and target for fold f
            train = pd.DataFrame(matrix_folds[0].toarray())
            train['target'] = matrix_folds[1]
            train['id_question'] = matrix_folds[2]
            train = train.reset_index()
            train = train.set_index('id_question')
            train = train.drop(columns=['index'])


            # Get test and target for fold f
            test = pd.DataFrame(matrix_folds[3].toarray())
            test['target'] = matrix_folds[4]
            test['id_question'] = matrix_folds[5]
            test = test.reset_index()
            test = test.set_index('id_question')
            test = test.drop(columns=['index'])

            #To get all dataset we just need to read the first fold (test and train)
            if f == 0:
                self.features = pd.concat([train, test])
                self.target = self.features['target']
                self.features = self.features.drop(columns = ['target'])

            # Given a id_question, get the position index for all questions
            train_indices.append(self.features.index.get_indexer_for(train.index))
            test_indices.append(self.features.index.get_indexer_for(test.index))

        self.cv = (list(zip(train_indices, test_indices)))


    # get each folds separately
    def get_separeted_folds(self):
        folds = []
        for f in range(0, self.cv.__len__()):
            for tt in range(0, 2):
                folds.append(self.features.iloc[self.cv[f][tt]].join(self.target, how = 'inner'))
        return folds

if __name__ == '__main__':
    #Just testing
    base = Dataset('cook')
