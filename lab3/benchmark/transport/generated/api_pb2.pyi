from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrainRequest(_message.Message):
    __slots__ = ("path",)
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...

class DataSet(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[Row]
    def __init__(self, rows: _Optional[_Iterable[_Union[Row, _Mapping]]] = ...) -> None: ...

class Benchmark(_message.Message):
    __slots__ = ("time", "mem")
    TIME_FIELD_NUMBER: _ClassVar[int]
    MEM_FIELD_NUMBER: _ClassVar[int]
    time: int
    mem: int
    def __init__(self, time: _Optional[int] = ..., mem: _Optional[int] = ...) -> None: ...

class Row(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: _containers.RepeatedScalarFieldContainer[float]
    y: float
    def __init__(self, x: _Optional[_Iterable[float]] = ..., y: _Optional[float] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("weights",)
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    weights: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, weights: _Optional[_Iterable[float]] = ...) -> None: ...

class TrainResponse(_message.Message):
    __slots__ = ("modelId", "benchmark")
    MODELID_FIELD_NUMBER: _ClassVar[int]
    BENCHMARK_FIELD_NUMBER: _ClassVar[int]
    modelId: str
    benchmark: Benchmark
    def __init__(self, modelId: _Optional[str] = ..., benchmark: _Optional[_Union[Benchmark, _Mapping]] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("modelId", "data")
    MODELID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    modelId: str
    data: DataSet
    def __init__(self, modelId: _Optional[str] = ..., data: _Optional[_Union[DataSet, _Mapping]] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("y", "benchmark")
    Y_FIELD_NUMBER: _ClassVar[int]
    BENCHMARK_FIELD_NUMBER: _ClassVar[int]
    y: _containers.RepeatedScalarFieldContainer[float]
    benchmark: Benchmark
    def __init__(self, y: _Optional[_Iterable[float]] = ..., benchmark: _Optional[_Union[Benchmark, _Mapping]] = ...) -> None: ...
