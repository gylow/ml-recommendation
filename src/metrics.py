import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


class Metrics:
    def __init__(self):
        pass

    def calculate_regression(self, y_true, y_pred):
        '''
        Calculate the metrics from a regression problem
        :param y_true: Numpy.ndarray or Pandas.Series
        :param y_pred: Numpy.ndarray or Pandas.Series
        :return: Dict with metrics
        '''
        median_abs_err = median_absolute_error(y_true, y_pred)
        mean_sqr_err = mean_squared_error(y_true, y_pred)
        r_2_score = r2_score(y_true, y_pred)
        return {'median_abs_err' : median_abs_err, 'mean_sqr_err' : mean_sqr_err, 'r_2_score' : r_2_score}
    