import logging
import os
import pathlib
import shutil
import zipfile
from pathlib import Path

import duckdb
import kaggle
import numpy
import pandas as pd
import scipy.sparse
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin


def rawdata(data: str = "train") -> duckdb.DuckDBPyRelation:
    root_dir = pathlib.Path(__file__).parent.parent
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



class Utils:
    @staticmethod
    def download(datadir: Path, competition: str, clean_first: bool = False) -> None:
        if clean_first:
            logging.info("Removing data directory: %s" % datadir.as_posix())
            shutil.rmtree(datadir.as_posix())
        if not datadir.joinpath("train.csv").exists():
            kaggle.api.authenticate()
            kaggle.api.competition_download_files(competition=competition, path=datadir.as_posix())
            with zipfile.ZipFile(datadir.joinpath("%s.zip" + competition).as_posix(), 'r') as zip_ref:
                zip_ref.extractall(datadir.as_posix())

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
