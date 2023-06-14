import os
import pathlib

import duckdb
import numpy
import pandas as pd
import scipy.sparse
from sklearn import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklego.preprocessing import RepeatingBasisFunction

from lib import Utils


def rawdata(data: str = "train") -> duckdb.DuckDBPyRelation:
    root_dir = pathlib.Path(__file__).parent
    os.chdir(root_dir)
    _data = duckdb.sql(" WITH h as ( "
                       " SELECT "
                       "   date,"
                       "   locale,"
                       "   locale_name, "
                       "   SUM(CASE WHEN type == 'Holiday' THEN 1 ELSE 0 END) as calendar_holiday,"
                       "   SUM(CASE WHEN type == 'Event' THEN 1 ELSE 0 END) as calendar_event,"
                       "   SUM(CASE WHEN type == 'Additional' THEN 1 ELSE 0 END) as calendar_additional,"
                       "   SUM(CASE WHEN type == 'Transfer' THEN 1 ELSE 0 END) as calendar_transfer,"
                       "   SUM(CASE WHEN type == 'Bridge' THEN 1 ELSE 0 END) as calendar_bridge,"
                       "   SUM(CASE WHEN transferred == true THEN 1 ELSE 0 END) as calendar_transferred,"
                       "   count(*)  AS num_holiday "
                       " FROM 'data/holidays_events.csv' "
                       " GROUP BY 1,2,3 "
                       " ) "
                       " SELECT "
                       "   t.id, "
                       "   t.date, "
                       # date feature extraction
                       "   CASE WHEN (last_day(t.date)==t.date OR datepart('day', t.date)==15) then 1 else 0 END AS payment_wages,"
                       "   t.store_nbr, "
                       "   t.family, "
                       "   %s "
                       "   t.onpromotion, "
                       "   s.city as store_city, "
                       "   s.state as store_state, "
                       "   s.type as store_type, "
                       "   s.cluster as store_cluster, "
                       "   o.dcoilwtico, "
                       "   tx.transactions, "
                       "   COALESCE(hl.calendar_holiday,hr.calendar_holiday,hn.calendar_holiday, 0) as calendar_holiday,"
                       "   COALESCE(hl.calendar_event,hr.calendar_event,hn.calendar_event, 0) as calendar_event,"
                       "   COALESCE(hl.calendar_additional,hr.calendar_additional,hn.calendar_additional, 0) as calendar_additional,"
                       "   COALESCE(hl.calendar_transfer,hr.calendar_transfer,hn.calendar_transfer, 0) as calendar_transfer,"
                       "   COALESCE(hl.calendar_bridge,hr.calendar_bridge,hn.calendar_bridge, 0) as calendar_bridge,"
                       "   COALESCE(hl.calendar_transferred,hr.calendar_transferred,hn.calendar_transferred, 0) as calendar_transferred,"
                       " FROM 'data/%s.csv' as t  "
                       " LEFT JOIN 'data/stores.csv' AS  s ON s.store_nbr = t.store_nbr "
                       " LEFT JOIN 'data/oil.csv' AS  o ON o.date = t.date "
                       " LEFT JOIN 'DATA/transactions.csv' AS tx ON t.date=tx.date AND t.store_nbr = tx.store_nbr \n"
                       " LEFT JOIN h AS hl ON t.date = hl.date AND hl.locale='Local' AND hl.locale_name=s.city \n"
                       " LEFT JOIN h AS hr ON t.date = hr.date AND hr.locale='Regional' AND hr.locale_name=s.state \n"
                       " LEFT JOIN h AS hn ON t.date = hn.date AND hn.locale='National' \n"
                       " ORDER BY t.store_nbr ASC, t.date ASC, t.family ASC "
                       % (" t.sales, " if data == "train" else "", data)
                       )
    assert _data.to_df().shape[0] == duckdb.sql("SELECT * FROM 'data/%s.csv' " % data).to_df().shape[0], "data duplicated!"
    return _data


class PrintTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if isinstance(X, numpy.ndarray):
            X_print = pd.DataFrame(X[1:5, :], columns=X.dtype.names)
            print(X_print.dtypes)
            print(X_print.to_markdown(tablefmt="grid"))
        elif isinstance(X, pd.DataFrame):
            print(X.dtypes)
            print(X.head(n=5).to_markdown(tablefmt="grid"))
        elif isinstance(X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            print(X.todense()[1:5, :])
        else:
            raise Exception("PrintTransformer unexpected type of X, %s" % type(X))
        return X


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
    df_test = rawdata(data="test").to_df()
    out = model.predict(df_test)
    predicted_test = y_scaler.inverse_transform(out)
    df_test['sales'] = predicted_test
    submit = df_test[['id', 'sales']]
    submit['sales'].loc[submit['sales'] < 0] = 0
    submit.to_csv(path_or_buf=DATA_DIR.joinpath("submit.csv"), index=False)
#     kaggle.api.competition_submit(file_name=DATA_DIR.joinpath("submit.csv").as_posix(), competition=COMPETITION, message="initial scikit model, FIRST UPLOAD")
