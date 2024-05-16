import functools

import grpc

from lab3.benchmark.transport.generated.api_pb2_grpc import MlStub
from transport.generated.api_pb2 import Model
import numpy as np


def get_stub():
    channel = grpc.insecure_channel("localhost:8888")
    return MlStub(channel)
