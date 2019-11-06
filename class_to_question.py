import string
import logging

import pandas as pd
from DBConnection import DB
import numpy as np
import os
from sklearn.datasets import dump_svmlight_file

if __name__ == '__main__':
    #For each base(cook, stack, english)
    for base_dir in filter(os.path.isdir, ['data/questionExtraction/' + s for s in os.listdir('data/questionExtraction')]):
        base = base_dir.split("/")[2]

        # Get the answer id and the answer's best view (which is also question's best view)
        feat_meta = pd.read_csv(base_dir + '/svm/Feat_Meta_' + base + '.csv')[
            ['best_view_result', 'id']].applymap(int)
        feat_meta = feat_meta.rename(columns={'id': 'answer_id'})

        # Get all questions and answers to join with feat_meat to retrieve question's best view by its answers
        bd = DB(base)
        q_a = pd.read_sql('SELECT DISTINCT id, parentId FROM Posts', bd.db).rename(columns={'id': 'answer_id'})

        # Join answer and question with answer_id to get the question_id (parentId) for each answer
        question = feat_meta.merge(q_a, left_on='answer_id', right_on='answer_id', how='inner').rename(
            columns={'parentId': 'question_id'})

        print(base)
        for f in range(0,5):
            for t in ['test', 'train']:
                print(base_dir + '/svm/class/class_fold_'+ str(f) +'_'+ t +'_all_.svm')
                fold = pd.read_csv(base_dir +'/svm/fold_'+ str(f) +'_'+ t +'_all_.svm', header=None, sep=" ", engine = 'python')
                fold = fold.loc[:, fold.columns != 0].applymap(lambda x: x.split(':')[1]).rename(columns = {89:'question_id'})
                fold.question_id = fold.question_id.apply(int)

                # Join fold with question to get the best view for each question
                fold = fold.merge(question, left_on='question_id', right_on='question_id', how='inner').drop(columns = ['answer_id'])

                label = fold.best_view_result
                question_id = fold.question_id
                fold = fold.drop(columns=['question_id', 'best_view_result'])

                if not os.path.exists(base_dir + '/svm/class'):
                    os.mkdir(base_dir + '/svm/class')

                dump_svmlight_file(fold, label, base_dir + '/svm/class/class_fold_'+ str(f) +'_'+ t +'_all_.svm', query_id=question_id)

                #fold.to_csv(base_dir + '/svm/class/class_fold_'+ str(f) +'_'+ t +'_all_.svm')





