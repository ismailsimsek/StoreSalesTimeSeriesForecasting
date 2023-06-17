import pathlib

import pandas as pd
from sklearn import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklego.preprocessing import RepeatingBasisFunction

from mymllib import Utils, rawdata


class CustomPreProcessingTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        if not isinstance(X, pd.DataFrame):
            raise Exception("CustomPreProcessingTransformer only supports pandas DataFrame input, type of X is: %s" % type(X))

        logger.info("Running CustomPreProcessingTransformer.transform method")
        X['store_nbr'] = 'store_' + X['store_nbr'].astype(str)
        X['store_cluster'] = 'store_cluster_' + X['store_cluster'].astype(str)
        X['transactions'] = X['transactions'].fillna(value=0)
        ####
        X['dcoilwtico'] = X['dcoilwtico'].ffill()
        X = X.drop(columns=["dcoilwtico"])
        # @TODO add number of stores in city
        # @TODO add number of stores in state
        logger.info("Null containing fields %s" % X.isna().any())
        # print(X.dtypes)
        #################### RepeatingBasisFunction ########################
        X['day_of_year'] = X["date"].dt.dayofyear
        X = X.drop(columns=["date"])
        #################### RepeatingBasisFunction ########################
        # TODO Choosing Fourier features with the Periodogram
        X_copy = X[['day_of_year']].copy(deep=True)
        rbt = RepeatingBasisFunction(n_periods=12, column="day_of_year", input_range=(1, 365), remainder="drop")
        rbt.fit(X_copy)
        X_copy = pd.DataFrame(index=X_copy.index, data=rbt.transform(X_copy))
        X_copy = X_copy.add_prefix("rbf_")
        X = pd.concat([X, X_copy], axis=1)
        X = X.drop(columns=["day_of_year"])
        #################### RepeatingBasisFunction ########################
        # print(X.dtypes)
        logger.info("Null containing fields %s" % X.isna().any())
        return X


if __name__ == '__main__':
    DATA_DIR = pathlib.Path(__file__).parent.joinpath("data")
    COMPETITION = 'store-sales-time-series-forecasting'

    Utils.download(datadir=DATA_DIR, competition=COMPETITION)
    df = rawdata(data="train").to_df()
    preprocessor = ColumnTransformer([
        ('one-hot-encoder', OneHotEncoder(sparse_output=False), make_column_selector(dtype_include=object)),
        # @TODO visualize transactions, and choose best scaler method by distribution
        ('standard-scaler', StandardScaler(), ["transactions"])
    ], remainder="passthrough", verbose_feature_names_out=True)
    pipe = Pipeline(steps=[
        ('customtransformer_preprocessor', CustomPreProcessingTransformer()),
        ('columntransformer_preprocessor', preprocessor),
        ('MultiOutputRegressor', MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=1000)))
    ])

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['sales']), df[['sales']], random_state=42)
    # @TODO visualize y, and choose best one by distribution
    # @TODO test log transformation

    # y_scaler = FunctionTransformer(func=np.log1p, inverse_func=np.exp)
    # y_scaler = MinMaxScaler()
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    # model = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    model = pipe
    model.fit(X_train, y_train)
    print('R2 score: {0:.2f}'.format(model.score(X_test, y_test)))
    ###################################################################################
    df_test = rawdata(data="test").to_df().drop(columns=['sales'])
    out = model.predict(df_test)
    predicted_test = y_scaler.inverse_transform(out)
    df_test['sales'] = predicted_test
    submit = df_test[['id', 'sales']]
    submit['sales'].loc[submit['sales'] < 0] = 0
    submit.to_csv(path_or_buf=DATA_DIR.joinpath("submit.csv"), index=False)
#     kaggle.api.competition_submit(file_name=DATA_DIR.joinpath("submit.csv").as_posix(), competition=COMPETITION, message="initial scikit model, FIRST UPLOAD")

