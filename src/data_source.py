import pandas as pd
from loguru import logger


class DataSource:

    def __init__(self,
                 name_id=None,
                 name_target=None,
                 rows_remove=None,
                 outliers_remove=None,
                 name_csv_label='label',
                 name_csv_train='train',
                 name_csv_test='test',
                 name_csv_predict='predict'):
        '''
        Deal data from data sources
        :param name_id: String with unique id column name.
        :param name_target: String with target column name in train dataframe.
        :param rows_remove: List of tuple(label, value_corresp) with the rows condictions to remove from Train data frame
        :param outliers_remove: List of tuple(label, value_corresp) with the rows condictions to remove from Train data frame
        :param name_csv_label: String with test target archive name without ".csv".
        :param name_csv_train: String with train archive name without ".csv".
        :param name_csv_test: String with test archive name without ".csv".
        :param name_csv_predict: String with predict archive name without ".csv".
        :return: DataSource object
        '''
        self.path_train = f'../data/{name_csv_train}.csv'
        self.path_test = f'../data/{name_csv_test}.csv'
        self.path_label = f'../data/{name_csv_label}.csv'
        self.path_predict = f'../data/{name_csv_predict}.csv'
        self.rows_remove = rows_remove
        self.outliers_remove = outliers_remove
        self.name_id = name_id
        self.name_target = name_target
        self.is_train_df = True
        self.data = None
        # TODO definir um seed padrão

    def set_df_train(self, path_original):
        '''
        Set train dataframe to work
        :param path_original: String with path from original file to train dataframe
        '''
        df_train = pd.read_csv(path_original, index_col=self.name_id)
        logger.success('Dataframe read')

        logger.info("Setting Train DataFrame")
        self.data = df_train
        self.is_train_df = True

        df_train.to_csv(self.path_train)
        logger.success('Writing train dataframe')

        logger.info(df_train.info(verbose=True))

    def set_target_by_index(self, serie_index):
        '''
        Set and write target column into train dataframe and make test dataframe without target column
        :param serie_index: Array like with index to classify
        '''
        if not self.name_target:
            self.name_target = 'target'

        if (self.data is None) or (not self.is_train_df):
            logger.info('Reading train dataframe')
            self.data = self.read_data()
            self.is_train_df = True
        else:
            logger.info(f"Train dataframe was already read")

        logger.info('Setting target column')
        self.data[self.name_target] = [
            1 if x in serie_index.to_numpy() else 0 for x in self.data.index]

        logger.info('Writing train dataframe with target column')
        self.data.to_csv(self.path_train)

        logger.info('Writing teste dataframe without index rows')
        self.data.drop(index=serie_index, columns=self.name_target).to_csv(
            self.path_test)
        logger.success('Dataframes was written')

    def set_train_columns_from_test(self):
        '''
        Set train dataframe columns equal from test dataframe
        '''
        if (self.data is None) or (not self.is_train_df):
            logger.info('Reading train dataframe')
            self.data = pd.read_csv(self.path_train, index_col=self.name_id)
            self.is_train_df = True
        else:
            logger.info(f"Train dataframe was already read")

        df_test = pd.read_csv(self.path_test, index_col=self.name_id)
        logger.success('Dataframes read')
        logger.info(
            f'Train shape: {self.data.shape} Test shape: {df_test.shape}')

        if self.name_target not in self.data.columns:
            logger.error('Train dataframe does not have target column')

        elif df_test.columns.size >= (self.data.drop(columns=self.name_target).columns.size)\
                and all(elem in df_test for elem in self.data.drop(columns=self.name_target).columns):
            logger.success('Test and train dataframes are ok')

        elif all(self.data[df_test.columns].columns == df_test.columns):
            logger.info(
                'Formating train dataframe columns equal test + target')
            self.data = pd.concat(
                [self.data[df_test.columns], self.data[self.name_target]], axis=1)
            logger.info('Writing formated train dataframe')
            self.data.to_csv('../data/train.csv',
                             index=(True if self.name_id else False))
            logger.success('Saved')
        else:
            logger.error('Test and train columns are totally differents')
        logger.info(self.data.info())

    def read_data(self, is_train_stage=True, original=False, usecols=None):
        '''
            Read data from data sources
            :param is_train_stage: Boolean specifing if is train or test.
            :param original: Boolean specifing if read original data frame or with removed rows
            :param usecols: List of name coluns to read 
            :return: pd.DataFrame with values and pd.Series with labels
        '''

        if (self.data is None) or (self.is_train_df != is_train_stage) or (original):
            logger.info(
                f"Reading {'train' if is_train_stage else 'test'} dataframe")
            self.data = pd.read_csv(
                self.path_train if is_train_stage else self.path_test, usecols=usecols, index_col=self.name_id).convert_dtypes()
            self.is_train_df = is_train_stage
        else:
            logger.info(
                f"{'Train' if is_train_stage else 'Test'} dataframe was already read")

        # The removed data can be recovered by the get_removed_rows()
        if self.rows_remove and not original:
            for label, x in self.rows_remove:
                self.data = self.data[self.data[label] != x]

        if is_train_stage and self.outliers_remove:
            for label, x in self.outliers_remove:
                self.data = self.data[self.data[label] != x]

        return self.data

    def get_removed_rows(self, name_columns=None, is_train_stage=True):
        '''
            Read especifics columns from data sources
            :param name_columns: List with columns names
            :param is_train_stage: Boolean specifing if is train or test.
            :return: pd.DataFrame with especificated columns
        '''

        if self.rows_remove is None:
            return pd.DataFrame(columns=[name_columns])

        df = self.read_data(is_train_stage, original=True)
        df = pd.concat([df[df[label] == x] for label, x in self.rows_remove])

        return df if name_columns is None else df[name_columns]

    def get_index(self, is_train_stage=True):
        '''
            Read index from data source
            :param is_train_stage: Boolean specifing if is train or test.
            :return: pd.Index 
        '''
        return self.read_data(is_train_stage, original=True, usecols=[0, 1]).index
