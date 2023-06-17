from unittest import TestCase

from mymllib import rawdata


class TestModel(TestCase):
    def test_rawdata(self):
        rawdata(data='train')
        rawdata(data='test')

    def test_duckdb(self):
        data = rawdata(data='test')
        print(data.columns)
        print(data.types)
        print(data.dtypes)
        print(type(data.columns))

    def test_save_load_model(self):
        raise NotImplemented

    def test_compare_new_model(self):
        raise NotImplemented
