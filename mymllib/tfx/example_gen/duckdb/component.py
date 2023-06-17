from typing import Optional, Union

from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2

from mymllib.tfx.example_gen.duckdb import executor


class DuckDBExampleGen(component.QueryBasedExampleGen):
    """Cloud DuckDBExampleGen component.
  
    The DuckDB examplegen component takes a query, and generates train
    and eval examples for downstream components.
  
    Component `outputs` contains:
     - `examples`: Channel of type `standard_artifacts.Examples` for output train
                   and eval examples.
    """

    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

    def __init__(
            self,
            query: Optional[str] = None,
            input_config: Optional[Union[example_gen_pb2.Input,
            data_types.RuntimeParameter]] = None,
            output_config: Optional[Union[example_gen_pb2.Output,
            data_types.RuntimeParameter]] = None,
            range_config: Optional[Union[range_config_pb2.RangeConfig,
            data_types.RuntimeParameter]] = None):
        """Constructs a DuckDBExampleGen component.
    
        Args:
          query: DuckDB sql string, query result will be treated as a single
            split, can be overwritten by input_config.
          input_config: An example_gen_pb2.Input instance with Split.pattern as
            DuckDB sql string. If set, it overwrites the 'query' arg, and allows
            different queries per split. If any field is provided as a
            RuntimeParameter, input_config should be constructed as a dict with the
            same field names as Input proto message.
          output_config: An example_gen_pb2.Output instance, providing output
            configuration. If unset, default splits will be 'train' and 'eval' with
            size 2:1. If any field is provided as a RuntimeParameter,
            input_config should be constructed as a dict with the same field names
            as Output proto message.
          range_config: An optional range_config_pb2.RangeConfig instance,
            specifying the range of span values to consider.
    
        Raises:
          RuntimeError: Only one of query and input_config should be set.
        """
        if bool(query) == bool(input_config):
            raise RuntimeError('Exactly one of query and input_config should be set.')
        input_config = input_config or utils.make_default_input_config(query)
        super().__init__(
            input_config=input_config,
            output_config=output_config,
            range_config=range_config)
