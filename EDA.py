from read_folds import load_base
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def feature_distribution(base):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.7)

    for f in base.keys()[0:40]:
        ax = fig.add_subplot(4, 10, f + 1)
        ax.tick_params(labelsize=3)
        ax.set_title('F' + str(f), fontsize=4)
        ax.hist(base[f])

    fig.savefig('data/img/0-40dist.pdf')

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6, wspace=0.7)

    i = 1
    for f in base.keys()[40:88]:
        ax = fig.add_subplot(5, 10, i)
        ax.set_title('F' + str(i), fontsize=4)
        ax.tick_params(labelsize=3)
        ax.hist(base[f])
        i += 1

    fig.savefig('data/img/40-88dist.pdf')


if __name__ == '__main__':
    base = load_base("cook")
    target = base['target']
    question_id = base['question_id']

    base = base.drop(columns = ['target', 'question_id'])

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = base.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = base.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])

    rf.fit(base, target)

    y_pred = rf.predict(base)
