import shutil
import unittest

import apache_beam as beam
import tensorflow as tf
import tfx.v1 as tfx

import tests
from mymllib.tfx.example_gen.duckdb.component import DuckDBExampleGen
from mymllib.tfx.example_gen.duckdb.executor import _DuckDBToExample


class Test(tests.BaseUnitTest):
    def test__duckdb_to_example(self):
        with beam.Pipeline() as pipeline:
            examples = (pipeline | 'ToTFExample' >> _DuckDBToExample(
                exec_properties={'_beam_pipeline_args': []},
                split_pattern='SELECT 321 as coll1int, True as coll2bool '
                              ' UNION ALL'
                              ' SELECT 222 as coll1int, False as coll2bool '))

            feature = {}
            feature['coll1int'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[321]))
            feature['coll2bool'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[True]))
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            self.assertEqual(example_proto, examples[0])
            self.assertEqual(2, len(examples))
            # util.assert_that(examples[0], util.equal_to([example_proto]))

    def test__duckdb_to_example_datetime(self):
        with beam.Pipeline() as pipeline:
            examples = (pipeline | 'ToTFExample' >> _DuckDBToExample(
                exec_properties={'_beam_pipeline_args': []},
                split_pattern="SELECT current_date as coll1date, current_timestamp as coll2tstz, "
                              " '1992-09-20 15:31:00+02'::TIMESTAMPTZ as  coll3tstz, '1992-09-20 15:30:00'::TIMESTAMP as  coll4ts "))

            self.assertIn('1992-09-20 15:30:00', str(examples))
            self.assertIn('1992-09-20 13:31:00', str(examples))

    @unittest.skip("manual run,test")
    def test_duckdb_example_gen(self):
        PIPELINE_PATH = self.TESTS_OUT.joinpath("pipeline").as_posix()
        METADATA_PATH = self.TESTS_OUT.joinpath("pipeline/metadata.db").as_posix()
        shutil.rmtree(PIPELINE_PATH)
        components = []
        sql = "select * from '%s' " % self.TESTS_RESOURCES.joinpath("csvdata/holidays_events.csv")
        print(sql)
        example_gen = DuckDBExampleGen(query=sql)
        components.append(example_gen)

        pipeline = tfx.dsl.Pipeline(
            pipeline_name="experimentpipeline",
            pipeline_root=PIPELINE_PATH,
            components=components,
            enable_cache=True,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH),
            # beam_pipeline_args=None,
        )
        tfx.orchestration.LocalDagRunner().run(pipeline)
