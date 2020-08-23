import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from experiments import Experiments
from joblib import dump, load
from loguru import logger

#from data_source import DataSource
#from preprocessing import Preprocessing


class ModelTraining:
    def __init__(self, preprocessing, regression=True, seed=42):
        self.pre = preprocessing
        self.regression = regression
        self.seed = seed
        '''
        :param preprocessing: Preprocessing object
        :param regression: Boolean representing the training model: True for Regression or False for Classification
        :param seed: Int with seed to random functions
        '''

    def training(self):
        '''
        Train the model.
        :return: Dict with trained model, preprocessing used and columns used in training
        '''

        logger.info("Training preprocessing")
        df_train, y = self.pre.process()

        logger.info("Training Model")
        exp = Experiments(regression=self.regression)
        df_metrics = exp.run_experiment(df_train, y, seed=self.seed)
        logger.info(f"Metrics: \n{df_metrics}")
        df_metrics.to_csv("../out/metrics.csv")

        if self.regression:
            alg_better = df_metrics[df_metrics.median_abs_err ==
                                    df_metrics.median_abs_err.min()].index[0]
        else:
            alg_better = df_metrics[df_metrics.r_2_score ==
                                    df_metrics.r_2_score.max()].index[0]
        logger.info(f"chosen algorithm: {alg_better}")

        model_obj = exp.get_model(alg_better)
        model_obj.fit(df_train, y)
        model = {'model_obj': model_obj,
                 'preprocessing': self.pre,
                 'colunas': self.pre.get_name_features()}

        # print(model)
        dump(model, '../out/model.pkl')

        return model
