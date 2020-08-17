import pandas as pd
import category_encoders as ce
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


class Preprocessing:
    def __init__(self, data, manual_feat=None, max_missing=0.0, max_unique=100, rfe=None, pca=None):
        self.features_categoric = None
        self.features_numeric = None
        self.scaler = None
        self.catb = None
        self.data = data
        self.rfe = rfe
        self.pca = pca
        self.manual_feat = manual_feat
        self.max_missing = max_missing
        self.max_unique = max_unique
        '''
        Class for preprocessing data model.
        :param data: DataSource object
        :param rfe: String with estimator: 'LR' or 'DT' for LinearRegression or DecisionTreeRegressor
        :param pca: Int, Float. Int for n_components to keep and Float for percetage specified by n_components
        :param manual_feat: List with: col_name : String,
                                        var_type : None, 'cat' or 'num'
                                        fillna : Int or Float,
                                        encode : Bool,
                                        drop_first : Bool
        :param max_missing: Float with maximum missing values acceptable percentage
        :param max_unique: Float with maximum unique values acceptable percentage
        :return: Preprocessed object
        '''

    def get_name_features(self):
        '''
        Get all features names for model
        :return: List[String] with categoric + numeric features names
        '''
        return self.features_numeric + self.features_categoric

    def get_name_target(self):
        '''
        Get target name from data source
        :return: String target name
        '''
        return self.data.name_target

    def _preprocess_manual(self, df):
        '''
        Manually preprocess dataframe setting categoric and numeric features
        :param df: Dataframe
        :return: Dataframe processed, List[String] with numeric features, List[String] with categoric features
        '''
        name, var_type, fill, encode, drop_first = 0, 1, 2, 3, 4
        features_numeric = list()
        features_categoric = list()

        for col in self.manual_feat:
            if col[var_type]:
                feature = features_categoric if col[var_type] == 'cat' else features_numeric
                print(f'use: {col[name]}')
                if col[fill] != None:
                    df[col[name]].fillna(col[fill], inplace=True)
                    print(f'\tfill na with: {col[fill]}')
                if col[encode]:
                    values = df[col[name]].value_counts(
                    ).sort_index().index.values
                    df = pd.get_dummies(df,
                                        columns=[col[name]],
                                        drop_first=col[drop_first],
                                        dtype='int')
                    print(f'\tencode: {values}')
                    if col[drop_first]:
                        print(f'\tdrop value: {values[0]}')
                        values = values[1:]
                    feature += [f'{col[name]}_{x}' for x in values]
                else:
                    feature.append(col[name])
            else:
                df.drop(columns=col[name], inplace=True)
                print(f'drop: {col[name]}')
        return df, features_numeric, features_categoric

    def _preprocess_auto(self, df):
        '''
        Automatically preprocess dataframe setting categoric and numeric features
        :param df: Dataframe
        :param missing_values_acceptable: Int with minimum missing values acceptable percentage
        :param unique_values_acceptable: Int with maximum unique values acceptable percentage
        :return: List[String] with numeric features, List[String] with categoric features
        '''

        features_categoric = df.select_dtypes(
            exclude='number').columns.to_list()
        features_numeric = df.select_dtypes(include='number').columns.to_list()

        logger.info('Creating DataFrame for Data Manipulation')
        percentage = 100/df.shape[0]
        df_meta = pd.DataFrame({'column': df.columns,
                                'missing_perc': df.isna().sum() * percentage,
                                'uniques': [df[x].value_counts().count() for x in df.columns],
                                'unique_perc': df.nunique() * percentage,
                                'dtype': df.dtypes})
        logger.info('\n'+ df_meta[['missing_perc', 'uniques',
                       'unique_perc', 'dtype']].round(3).to_string())

        logger.info(
            f'Droping columns with unique values >= {self.max_unique}% :')
        logger.info(
            f'Droping columns with missing values > {self.max_missing}% :')
        to_del = df_meta[(df_meta['missing_perc'] > self.max_missing) | (
            df_meta['unique_perc'] >= self.max_unique)].column
        for x in to_del:
            if x in features_categoric:
                features_categoric.remove(x)
            else:
                features_numeric.remove(x)

        if self.data.name_target in features_numeric:
            features_numeric.remove(self.data.name_target)
        if self.data.name_target in features_categoric:
            features_categoric.remove(self.data.name_target)

        return features_numeric, features_categoric

    def _select_features(self, df, y, feat_num, feat_cat):
        '''
        Select features from train dataframe
        :param df: Dataframe
        :param y: Serie with target values
        :param feat_num: List[String] with unselected numeric features
        :param feat_cat: List[String] with unselected categoric features
        :return: List[String] with selected numeric features, List[String] with selected categoric features
        '''

        df[feat_num] = StandardScaler().fit_transform(df[feat_num])
        df[feat_cat] = ce.CatBoostEncoder(cols=feat_cat).fit_transform(
            df[feat_cat], y=y)

        if self.pca:
            pca = PCA(self.pca).fit(df[feat_num+feat_cat])
            n_components = pca.n_components_
            logger.info(
                f"Numero de componentes selecionados PCA é {n_components}")
        else:
            n_components = int(len(feat_num+feat_cat)/2)

        model = DecisionTreeRegressor() if self.rfe == 'DT' else LinearRegression()

        selection = RFE(model, n_features_to_select=n_components)
        selection.fit(df[feat_num+feat_cat], y=y)
        feat_selected = df[feat_num +
                           feat_cat].columns[selection.get_support()]

        for feat in feat_num:
            if feat not in feat_selected:
                feat_num.remove(feat)

        for feat in feat_cat:
            if feat not in feat_selected:
                feat_cat.remove(feat)

        return feat_num, feat_cat

    def _process_train(self, df, feat_num, feat_cat):
        '''
        Process train dataframe with fit and transform
        :param df: Dataframe
        :param feat_num: List[String] with numeric features
        :param feat_cat: List[String] with categoric features
        :return: Fitted and transformed dataframe, and target serie
        '''
        # TODO implementar testes automatizados para garantir que os dados de x e y continuam correspondentes

        logger.info(
            'Setting Y as target and Removing target from train dataframe')
        y = df[self.get_name_target()].fillna(0)
        df = df.drop(columns={self.get_name_target()})

        if self.rfe:
            logger.info('Select features')
            self._select_features(df.copy(), y, feat_num, feat_cat)
            logger.info(f'Numeric Feature Selected >>>> {feat_num}')
            logger.info(f'Categoric Feature Selected >>>> {feat_cat}')

        logger.info('Feature fit and transform in train dataframe')
        self.scaler = StandardScaler()
        self.catb = ce.CatBoostEncoder(cols=feat_cat)

        df[feat_num] = self.scaler.fit_transform(df[feat_num])
        df[feat_cat] = self.catb.fit_transform(df[feat_cat], y=y)

        self.features_numeric = feat_num
        self.features_categoric = feat_cat

        return df[self.get_name_features()], y

    def _process_test(self, df):
        '''
        Process train dataframe with transform
        :param df: Dataframe
        :return: Transformed dataframe
        '''
        print('Feature Transform in test dataframe')
        df[self.features_numeric] = self.scaler.transform(
            df[self.features_numeric])
        df[self.features_categoric] = self.catb.transform(
            df[self.features_categoric])

        return df[self.get_name_features()]

    def process(self, is_train_stage=True):
        '''
        Process data for training the model.
        :param etapa_treino: Boolean
        :return: processed Pandas Data Frame, and target if train stage
        '''
        if not self.get_name_target:
            logger.error('Target name not defined')
        df = self.data.read_data(is_train_stage)

        if self.manual_feat:
            df, feat_num, feat_cat = self._preprocess_manual(df)
        else:
            feat_num, feat_cat = self._preprocess_auto(df)

        if is_train_stage:
            logger.info(f'Numeric Feature >>>> {feat_num}')
            logger.info(f'Categoric Feature >>>> {feat_cat}')
            return self._process_train(df, feat_num, feat_cat)
        else:
            logger.info(f'Numeric Feature >>>> {self.features_numeric}')
            logger.info(f'Categoric Feature >>>> {self.features_categoric}')
            return self._process_test(df)
