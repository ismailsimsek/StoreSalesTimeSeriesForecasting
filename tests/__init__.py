import unittest
from pathlib import Path


class BaseUnitTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)
        self.TESTS = Path(__file__).parent
        self.TESTS_OUT = self.TESTS.joinpath("out")
        self.TESTS_RESOURCES = self.TESTS.joinpath("resources")


def startTestRun(self):  # pylint: disable=unused-argument
    """
    https://docs.python.org/3/library/unittest.html#unittest.TestResult.startTestRun
    Called once before any tests are executed. initializes global unit test resources before starting tests

    :return:
    """
    pass  # pylint: disable=unnecessary-pass


setattr(unittest.TestResult, 'startTestRun', startTestRun)


def stopTestRun(self):  # pylint: disable=unused-argument
    """
    https://docs.python.org/3/library/unittest.html#unittest.TestResult.stopTestRun
    Called once after all tests are executed. removes/cleans global unit test resources post test run

    :return:
    """
    pass  # pylint: disable=unnecessary-pass
    # logging.warning("Removing output dir after post test")
    # shutil.rmtree(BaseUnitTest().TESTS_OUT.as_posix())


setattr(unittest.TestResult, 'stopTestRun', stopTestRun)
