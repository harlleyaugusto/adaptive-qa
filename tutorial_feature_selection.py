import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from feature_selection import univariate_selection
from pipe_algorithms import pipele_classification
from read_folds import load_folds, load_base

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

def zscore(base):
    base_norm = (base - base.mean()) / base.std()
    for feat in range(0, base.columns.__len__(), 1):
        base_plot = pd.concat([target, base_norm.iloc[:, feat:(feat + 1)]], axis=1)
        base_plot = pd.melt(base_plot, id_vars="target", var_name="features", value_name='value')
        plt.figure(figsize=(15, 15))
        sns.violinplot(x="features", y="value", hue="target", data=base_plot, inner='quartile')
        # plt.show()
        plt.savefig('data/img/zscore/' + base_name + '_' + 'z_score_feat_' + str(feat) + '-' + str(feat + 1) + '.png')

def swarm(base):
    base_norm = (base - base.mean()) / base.std()
    for feat in range(0, base.columns.__len__(), 10):
        base_plot = pd.concat([target, base_norm.iloc[:, feat:(feat + 10)]], axis=1)
        base_plot = pd.melt(base_plot, id_vars="target",
                       var_name="features",
                       value_name='value')
        plt.figure(figsize=(8, 8))
        sns.swarmplot(x="features", y="value", hue="target", data=base_plot)

        plt.xticks(rotation=90)

        plt.savefig('data/img/swarm/' + base_name + '_' + 'swarm_feat_' + str(feat) +'-'+ str(feat+10) +'.png')

def corrAll(base):
    f, ax = plt.subplots(figsize=(20, 20))
    sns.set(font_scale=0.4)
    sns.heatmap(base.iloc[:, 0:88].corr(), square=True,linewidths=.2, fmt='.1f', ax=ax, annot=True)
    plt.savefig('data/img/' + base_name + '_' + 'correlation.pdf')

    c = base.corr().abs()
    s = c.unstack().sort_values()
    s = s[s >= 0.9].index

    print(' \n'.join(map(str, s)))
    return s

if __name__ == '__main__':
    base_name = 'cook'
    base = load_base(base_name)

    base.head()  # head method show only first 5 rows
    target = base['target']
    base.drop(columns = ['target', 'question_id'], inplace = True)


    # feature names as a list
    col = base.columns  # .columns gives columns names in data
    print(col)

    #Drop some features here. Using covariance here, probably.

   #ax = sns.countplot(target, label="Count")
   # plt.show()

    #swarm(base)
    #corrAll(base)


    # drop based on the correlation table
    drop_list = [46, 42, 7, 23, 14, 32, 35, 24, 34, 40, 34, 20, 10, 7, 3, 45]

    base_2 = base.drop(columns = drop_list)

    base = 'cook'
    folds = load_folds(base)

    for f in range(0, folds.__len__()):
        folds[f] = folds[f].drop(columns= drop_list)

    folds_trans = univariate_selection(folds, k=5)

    pipele_classification(folds_trans)





