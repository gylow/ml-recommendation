import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from metrics import Metrics
from loguru import logger
#from sklearn.linear_model import RidgeCV
#from sklearn.linear_model import LassoCV
#from sklearn.model_selection import GridSearchCV
#from RandomForestClassifier
#from sklearn.neighbors import NearestNeighbors


class Experiments:
    def __init__(self, regression=True):
        self.regression_algorithms = {'LinearRegression': LinearRegression(),
                                      'Ridge': Ridge(),
                                      'Lasso': Lasso(),
                                      'DecisionTreeRegressor': DecisionTreeRegressor(),
                                      'RandomForestRegressor': RandomForestRegressor(),
                                      'SVR': SVR(),
                                      'CatBoostRegressor': CatBoostRegressor()
                                      }
        self.classification_algorithms = {'DecisionTreeRegressor': DecisionTreeRegressor(),
                                          'RandomForestRegressor': RandomForestRegressor(),
                                          'CatBoostRegressor': CatBoostRegressor(),
                                          'KNeighborsClassifier': KNeighborsClassifier(),
                                          'LogisticRegression': LogisticRegression()}
        self.dict_of_models = None
        self.regression = regression
        '''
        Choose the best algorithms to fit the problem
        :param regression: Boolean representing the training model: True for Regression or False for Classification
        '''
        # TODO implementar hiperparametros

    def get_model(self, alg):
        '''
        :param alg: String with the algorithm name to return
        :return: Algorithm class
        '''
        if self.regression:
            return self.regression_algorithms[alg]
        else:
            return self.classification_algorithms[alg]

    def train_model(self, x_train, y_train):
        '''
        Train the model with especified experiments
        :param x_train: pd.DataFrame with train data
        :param y_train: pd.Series with train labels
        :return: Dict with trained model
        '''

        algorithms = self.regression_algorithms if self.regression else self.classification_algorithms

        for alg in algorithms.keys():
            test = algorithms[alg]

            logger.info(test)
 
            test.fit(x_train, y_train)
            
            if self.dict_of_models is None:
                self.dict_of_models = {alg: test}
            else:
                self.dict_of_models.update({alg: test})
        return self.dict_of_models

    def run_experiment(self, df, y, test_size=0.15, seed=42):
        '''
        Run especified experiments
        :param df: Data Frame with features and target
        :param test_size: Float with percentage splited to test
        :param seed_random: Int with seed to random functions
        :return: Dataframe with all metrics
        '''

        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)
        
        models = self.train_model(x_train, y_train)
        
        df_metrics = pd.DataFrame()

        logger.info("Running Metrics")
        for model in models.keys():
            logger.info(f"Predizendo os testes de {model}")
            y_pred = models[model].predict(x_test)

            logger.info(f"y : {np.sort(y_pred)[-10:].round(2)}")

            metrics = Metrics().calculate_regression(y_test, pd.Series(y_pred))            
            df_metrics = df_metrics.append(pd.Series(metrics, name=model))

            pd.DataFrame.from_dict(metrics, orient='index').to_csv(
                '../out/'+model+'.csv')

        return df_metrics
