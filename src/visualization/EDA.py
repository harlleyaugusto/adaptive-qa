import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

import src.data.Dataset as dt
from src.features.preprocess_features import PreprocessFeatures
import src.util.config as c

def feature_distribution(base):
    config = c.Config()
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.7)

    for f in base.features.keys()[0:40]:
        ax = fig.add_subplot(4, 10, f + 1)
        ax.tick_params(labelsize=3)
        ax.set_title('F' + str(f), fontsize=4)
        ax.hist(base.features[f])

    fig.savefig(config.report + config.img_report + '0-40dist.pdf')

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6, wspace=0.7)

    i = 1
    for f in base.features.keys()[40:88]:
        ax = fig.add_subplot(5, 10, i)
        ax.set_title('F' + str(i), fontsize=4)
        ax.tick_params(labelsize=3)
        ax.hist(base.features[f])
        i += 1

    fig.savefig(config.report + config.img_report + '40-88dist.pdf')


if __name__ == '__main__':
    base = dt.Dataset("cook")
    prepro = PreprocessFeatures(base)

    rf = Pipeline(steps=[('preprocessor', prepro.get_preprocessor()), ('classifier', RandomForestClassifier())])

    y_pred = cross_val_predict(rf, base.features, base.target, cv = base.cv , verbose = 100)

    feature_distribution(base)

    #y_pred = rf.predict(base)
