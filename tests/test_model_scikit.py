from unittest import TestCase

from model_scikit import rawdata


class TestModel(TestCase):
    def test_rawdata(self):
        rawdata(data='train')
        rawdata(data='test')
