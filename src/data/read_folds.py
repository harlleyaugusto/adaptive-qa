import pandas as pd
from src.data.DBConnection import DB
import os
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_files
import src.util.config as c

def get_best_view_question():
    config = c.Config()

    for b in config.datasets:
        base = b.strip()

        # Get the answer id and the answer's best view (which is also question's best view)
        feat_meta = pd.read_csv(config.data_dir + config.raw_data + base +'/svm/Feat_Meta_' + base + '.csv')[
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
                print(config.processed_data + base + '/class_fold_' + str(f) + '_' + t + '_all_.svm')
                fold = pd.read_csv(config.data_dir + config.raw_data + base + '/svm/fold_' + str(f) + '_' + t + '_all_.svm', header=None, sep=" ",
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

                if not os.path.exists(config.data_dir + config.processed_data + base):
                    os.mkdir(config.data_dir + config.processed_data + base)

                dump_svmlight_file(fold, label, config.data_dir + config.processed_data + base +'/'+ 'class_fold_' + str(f) + '_' + t + '_all_.svm',
                                   query_id=question_id)


if __name__ == '__main__':
    get_best_view_question()



