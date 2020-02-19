import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import src.util.config as c
import src.data.Dataset as dt

def zscore(base):
    config = c.Config()
    base_norm = (base.features - base.features.mean()) / base.features.std()
    for feat in range(0, base.features.columns.__len__(), 1):
        base_plot = pd.concat([base.target, base_norm.iloc[:, feat:(feat + 1)]], axis=1)
        base_plot = pd.melt(base_plot, id_vars="target", var_name="features", value_name='value')
        plt.figure(figsize=(15, 15))
        sns.violinplot(x="features", y="value", hue="target", data=base_plot, inner='quartile')
        # plt.show()

        plt.savefig(config.report + config.img_report + base.name + '_' + 'z_score_feat_' + str(feat) + '-' + str(feat + 1) + '.png')

def swarm(base):
    config = c.Config()
    base_norm = (base.features - base.features.mean()) / base.features.std()
    for feat in range(0, base.features.columns.__len__(), 10):
        base_plot = pd.concat([base.target, base_norm.iloc[:, feat:(feat + 10)]], axis=1)
        base_plot = pd.melt(base_plot, id_vars="target",
                       var_name="features",
                       value_name='value')
        plt.figure(figsize=(8, 8))
        sns.swarmplot(x="features", y="value", hue="target", data=base_plot)

        plt.xticks(rotation=90)

        plt.savefig(config.report + config.img_report + base.name + '_' + 'swarm_feat_' + str(feat) +'-'+ str(feat+10) +'.png')

def corrAll(base):
    config = c.Config()
    f, ax = plt.subplots(figsize=(20, 20))
    sns.set(font_scale=0.4)
    sns.heatmap(base.features.iloc[:, 0:88].corr(), square=True,linewidths=.2, fmt='.1f', ax=ax, annot=True)
    plt.savefig(config.report + config.img_report + base.name + '_' + 'correlation.pdf')

    corr = base.features.corr().abs()
    s = corr.unstack().sort_values()
    s = s[s >= 0.9].index

    print(' \n'.join(map(str, s)))
    return s

if __name__ == '__main__':
    base = dt.Dataset('cook')

    #swarm(base)
    #zscore(base)
    corrAll(base)



