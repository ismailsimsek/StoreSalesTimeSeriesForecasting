from typing import Any, Dict

import apache_beam as beam
import duckdb
import tensorflow as tf
from absl import logging
from tfx.components.example_gen import base_example_gen_executor

from mymllib import ThreadSafeCounter


class _DuckDBConverter:
    """Help class for DuckDB result row to tf example conversion."""

    def __init__(self, query: str):
        """Instantiate a _DuckDBConverter object.
    
        Args:
          query: the query statement to get the type information.
        """

        # Dummy query to get the type information for each field.
        results: duckdb.DuckDBPyRelation = duckdb.query('SELECT * FROM ({}) LIMIT 0'.format(query))
        result_fields = dict(zip(results.columns, results.dtypes))
        self._type_map = {}

        for field_name, field_type in result_fields.items():
            self._type_map[field_name] = field_type

    def RowToExample(self, instance: Dict[str, Any]) -> tf.train.Example:
        """Convert DuckDB result row to tf example.        
        Args:
          field_to_type: The name of the field to its type from DuckDB.
          field_name_to_data: The data need to be converted from DuckDB that
            contains field name and data.
        
        Returns:
          A tf.train.Example that converted from the DuckDB row. Note that BOOLEAN
          type in DuckDB result will be converted to int in tf.train.Example.
        
        Raises:
          RuntimeError: If the data type is not supported to be converted.
            Only INTEGER, BOOLEAN, FLOAT, STRING is supported now.
        """
        feature = {}
        for key, value in instance.items():
            data_type = self._type_map[key]

            if value is None:
                feature[key] = tf.train.Feature()
                continue

            value_list = value if isinstance(value, list) else [value]
            if data_type in ('INTEGER', 'BOOLEAN', 'BIGINT', 'TINYINT', 'SMALLINT', 'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT'):
                feature[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value_list))
            # elif data_type in ('HUGEINT'):
            #     feature[key] = tf.train.Feature(
            #         int128_list=tf.train.Int128List(value=value_list))
            elif data_type in ('DATE', 'TIMESTAMP WITH TIME ZONE', 'TIMESTAMP', 'TIME'):
                feature[key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.compat.as_bytes(str(elem)) for elem in value_list]))
            elif data_type in ('FLOAT', 'DOUBLE'):
                feature[key] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value_list))
            elif data_type == 'VARCHAR':
                feature[key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.compat.as_bytes(elem) for elem in value_list]))
            else:
                # TODO: support HUGEINT types.
                raise RuntimeError('DuckDB column "{}" has non-supported type {} value {}.'.format(key, data_type, value_list))

        return tf.train.Example(features=tf.train.Features(feature=feature))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _DuckDBToExample(pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
                     split_pattern: str) -> beam.pvalue.PCollection:
    """Read from DuckDB and transform to TF examples.
  
    Args:
      pipeline: beam pipeline.
      exec_properties: A dict of execution properties.
      split_pattern: Split.pattern in Input config, a DuckDB sql string.
  
    Returns:
      PCollection of TF examples.
    """

    def readFromDuckDB(query: str):
        resultset = duckdb.sql(query=query)
        resultset_columns = resultset.columns
        resultlist: list = resultset.fetchall()
        num_rows = len(resultlist)
        COUNTER = ThreadSafeCounter()
        for row in resultlist:
            COUNTER.increment()
            if COUNTER.counter % 10000 == 0:
                logging.debug("Processed: %s out of %s rows" % (COUNTER.counter, num_rows))
            yield dict(zip(resultset_columns, row))

    converter = _DuckDBConverter(split_pattern)

    return (pipeline
            | beam.Create([split_pattern])
            | 'ReadFromDuckDB' >> beam.FlatMap(readFromDuckDB)
            | 'ToTFExample' >> beam.Map(converter.RowToExample))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
    """Generic TFX DuckDBExampleGen executor."""

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for DuckDB to TF examples."""
        return _DuckDBToExample
