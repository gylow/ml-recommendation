import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from IPython.core.pylabtools import figsize
from loguru import logger


class Visualization:
    def __init__(self, df):
        '''
        :param df: Dataframe to visualization
        '''
        self.df = df
        # %matplotlib inline
        figsize(12, 8)
        sns.set()

    def regression_viz(self, y_true, y_pred, nome):
        '''
        Visualize the quality of regression model
        :param y_true: pd.Series with true label values
        :param y_pred: pd.Series with predicted label values
        :param nome: Name of the file wich will be saved
        :return: Save files in specified path
        '''
        residual = y_pred - y_true
        data = pd.DataFrame(
            {'pred': y_pred, 'true': y_true, 'residual': residual})
        plot1 = sns.distplot(data['residual'], bins=50)
        plot2 = sns.scatterplot(x='true', y='residual', data=data)
        plt.savefig(plot1, '../data/'+nome+'_distplot.csv')
        plt.savefig(plot2, '../data/' + nome + 'scatterplot.csv')
        plt.show()

    def feature_analise(self, name_feat, name_target=None, y=None):
        '''
        Unique feature visual analise 
        :param name_feat: String with column name to analise
        :param name_target: String with target column name
        :param y: Series with target values
        '''

        if not pd.StringDtype().is_dtype(self.df[name_feat]):
            logger.info('Discrete feature')
            sns.distplot(self.df[name_feat])
            plt.show()
            sns.boxplot(self.df[name_feat])
            plt.show()
            if not (self.df[name_feat].hasnans) and\
                    not (pd.BooleanDtype().is_dtype(self.df[name_feat])):
                logger.info('Hasn\'t nulls values')
                sm.qqplot(self.df[name_feat], fit=True, line="45")
                plt.show()

        if name_target or (y is not None):
            logger.info('With target feature')
            if name_target:
                y = self.df[name_target]
            sns.lineplot(self.df[name_feat], y)
            plt.show()
            sns.scatterplot(self.df[name_feat], y)
            plt.show()

    def features_corralations(self):
        sns.heatmap(self.df.corr(), square=True)
