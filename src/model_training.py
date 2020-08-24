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

        if True:  # TODO fix this metrics
            alg_better = df_metrics.nsmallest(
                1, "mean_sqr_err").index[0]
        elif self.regression:
            alg_better = df_metrics.nsmallest(
                1, ["median_abs_err", "mean_sqr_err"]).index[0]
        else:
            alg_better = df_metrics.nlargest(1, "r_2_score").index[0]
        logger.info(f"chosen algorithm: {alg_better}")

        model_obj = exp.get_model(alg_better)
        model_obj.fit(df_train, y)
        model = {'model_obj': model_obj,
                 'preprocessing': self.pre,
                 'colunas': self.pre.get_name_features()}

        dump(model, '../out/model.pkl')

        return model
