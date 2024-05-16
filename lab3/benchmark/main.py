from transport.generated.api_pb2_grpc import MlStub
from transport.generated.api_pb2 import PredictRequest, TrainRequest, DataSet, Row
import grpc

channel = grpc.insecure_channel("localhost:8888")
s = MlStub(channel)
# r = s.train(TrainRequest(path="./test-data/test.csv"))
# print(r)
r = s.predict(PredictRequest(modelId="d1c446b1-b994-4b19-a3fe-751812d73576", data=DataSet(
    rows=[
        Row(x=[-0.29451304412658563, -0.49304650406076056]),
        *[Row(x=[-0.3207816957162245, 0.4676085387753295])] * 100000,
    ]
)))
print(r)
