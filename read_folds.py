import pandas as pd
from DBConnection import DB
import os
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_files
import matplotlib.pyplot as plt

def get_best_view_question():
    # For each base(cook, stack, english)
    for base_dir in filter(os.path.isdir,
                           ['data/questionExtraction/' + s for s in os.listdir('data/questionExtraction')]):
        base = base_dir.split("/")[2]

        # Get the answer id and the answer's best view (which is also question's best view)
        feat_meta = pd.read_csv(base_dir + '/svm/Feat_Meta_' + base + '.csv')[
            ['best_view_result', 'id']].applymap(int)
        feat_meta = feat_meta.rename(columns={'id': 'answer_id'})

        # Get all questions and answers to join with feat_meat to retrieve question's best view by its answers
        db = DB(base)
        q_a = pd.read_sql('SELECT DISTINCT id, parentId FROM Posts', db.db)

        # Join answer and question with answer_id to get the question_id (parentId) for each answer
        question = feat_meta.merge(q_a, left_on='answer_id', right_on='id', how='inner').rename(
            columns={'parentId': 'question_id'})

        print(base)
        for f in range(0, 5):
            for t in ['test', 'train']:
                print(base_dir + '/svm/class/class_fold_' + str(f) + '_' + t + '_all_.svm')
                fold = pd.read_csv(base_dir + '/svm/fold_' + str(f) + '_' + t + '_all_.svm', header=None, sep=" ",
                                   engine='python')
                fold = fold.loc[:, fold.columns != 0].applymap(lambda x: x.split(':')[1]).rename(
                    columns={89: 'question_id'})
                fold.question_id = fold.question_id.apply(int)

                # Join fold with question to get the best view for each question
                fold = fold.merge(question, left_on='question_id', right_on='question_id', how='inner').drop(
                    columns=['id'])

                label = fold.best_view_result
                question_id = fold.question_id
                fold = fold.drop(columns=['question_id', 'best_view_result', 'answer_id'])

                if not os.path.exists(base_dir + '/svm/class'):
                    os.mkdir(base_dir + '/svm/class')

                dump_svmlight_file(fold, label, base_dir + '/svm/class/class_fold_' + str(f) + '_' + t + '_all_.svm',
                                   query_id=question_id)

#
#  Read the base given by base_name (i.e read all class_fold_*_test_all_.svm files), and return it as a
#  dataframe
#

def load_base(base_name):
    base_path = 'data/questionExtraction/' + base_name + "/svm/class/"

    #Read all test (i.e. read the entire base)
    files = list(filter(lambda x: str.endswith(x,"test_all_.svm"), os.listdir(base_path)))

    folds = load_svmlight_files([ base_path + s for s in files], query_id = True)
    data = []

    for f in range(0, folds.__len__(),3):
        a = pd.DataFrame(folds[f].toarray())
        a['target'] = folds[f+1]
        a['question_id'] = folds[f+2]
        data.append(a)

    data = pd.concat(data).reset_index();
    return data.drop(columns = ['index'])

def load_folds_old(base_name, fold_num = None):
    base_path = 'data/questionExtraction/' + base_name + "/svm/class/"
    folds = []

    for f in (range(0,5) if fold_num is None or fold_num.__len__() == 0 else fold_num):
        files = [base_path + '/class_fold_' + str(f) + '_train_all_.svm']
        files.append(base_path + '/class_fold_' + str(f) + '_test_all_.svm')

        matrix_folds = load_svmlight_files(files)

        # Get train and target
        a = pd.DataFrame(matrix_folds[0].toarray())
        a['target'] = matrix_folds[1]
        folds.append(a)

        # Get test and target
        a = pd.DataFrame(matrix_folds[2].toarray())
        a['target'] = matrix_folds[3]
        folds.append(a)

    return folds

def load_folds(base_name, fold_num = None):
    base_path = 'data/questionExtraction/' + base_name + "/svm/class/"
    folds = []

    for f in (range(0,5) if fold_num is None or fold_num.__len__() == 0 else fold_num):
        files = [base_path + '/class_fold_' + str(f) + '_train_all_.svm']
        files.append(base_path + '/class_fold_' + str(f) + '_test_all_.svm')

        matrix_folds = load_svmlight_files(files, query_id= True)

        # Get train and target
        a = pd.DataFrame(matrix_folds[0].toarray())
        a['target'] = matrix_folds[1]
        a['id_query'] = matrix_folds[2]
        a = a.reset_index()
        a = a.set_index('id_query')
        a = a.drop(columns=['index'])
        folds.append(a)

        # Get test and target
        a = pd.DataFrame(matrix_folds[3].toarray())
        a['target'] = matrix_folds[4]
        a['id_query'] = matrix_folds[5]
        a = a.reset_index()
        a = a.set_index('id_query')
        a = a.drop(columns=['index'])
        folds.append(a)

    return folds

if __name__ == '__main__':
    folds = load_folds('cook')

    '''
    get_best_view_question()
    base = load_base("cook")
    base['target'].value_counts().sort_index().plot(kind='bar')
    plt.title("cook")
    plt.show()

    base = load_base("english")
    base['target'].value_counts().sort_index().plot(kind='bar')
    plt.title("english")
    plt.show()

    base = load_base("stack")
    base['target'].value_counts().sort_index().plot(kind='bar')
    plt.title("Stack")
    plt.show()

    '''




