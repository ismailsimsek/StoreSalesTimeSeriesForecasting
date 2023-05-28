import pathlib
import pprint

import tensorflow as tf
from absl import logging
from tfx import v1 as tfx
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.proto import example_gen_pb2
from tfx.v1 import proto

from mymllib import Utils, rawdata_duckdb_sql
from mymllib.tfx.example_gen.duckdb.component import DuckDBExampleGen

pp = pprint.PrettyPrinter()
logging.set_verbosity(logging.INFO)  # Set default logging level.

if __name__ == '__main__':
    ###################################################################################################################
    print('TensorFlow version: {}'.format(tf.__version__))
    print('TFX version: {}'.format(tfx.__version__))
    ########################### Set up variables ####################################################################
    COMPETITION = 'store-sales-time-series-forecasting'
    PIPELINE_NAME = COMPETITION
    CURRENT_DIR: pathlib.Path = pathlib.Path(globals()['_dh'][0]) if '_dh' in globals() else pathlib.Path(__file__).parent
    DATA_DIR = CURRENT_DIR.joinpath("data")
    TFX_DIR = CURRENT_DIR.joinpath("out/tfx").joinpath(PIPELINE_NAME)
    PIPELINE_ROOT = TFX_DIR.joinpath('pipelines')
    METADATA_PATH = TFX_DIR.joinpath('metadata.db')
    SERVING_MODEL_DIR = TFX_DIR.joinpath('serving_model')
    ###########################Set up variables#####################################################################
    Utils.download(datadir=DATA_DIR, competition=COMPETITION)
    components = []
    output = proto.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            proto.SplitConfig.Split(name='train', hash_buckets=8),
            proto.SplitConfig.Split(name='eval', hash_buckets=2)
        ]))
    example_gen = DuckDBExampleGen(query=rawdata_duckdb_sql(data="train"), output_config=output)
    components.append(example_gen)
    ## INFO:absl:Publishing output artifacts 
    # # defaultdict(<class 'list'>, {'examples': [Artifact(artifact: uri: "/Users/ismailsimsek/development/StoreSalesTimeSeriesForecasting/out/tfx/store-sales-time-series-forecasting/pipelines/DuckDBExampleGen/examples/3"
    # Computes statistics over data for visualization and example validation.
    print(example_gen.outputs['examples'])
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)
    # Generates schema based on statistics files.
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    # components.append(schema_gen)
    ####################################################################################################################################
    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    # components.append(example_validator)
    ###################### PIPELINE ####################################################################################################
    Utils.remove_files(dir=TFX_DIR, file_pattern=".DS_Store")
    metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH.as_posix())
    pipeline = tfx.dsl.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT.as_posix(),
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_connection_config
    )
    result = tfx.orchestration.LocalDagRunner().run(pipeline)
