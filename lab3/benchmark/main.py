from lab3.benchmark.dataset_generator import generate, GeneratorConfig, Functions
from lab3.benchmark.model import get_stub
from lab3.benchmark.visualization import plot_model_over_dataset
from transport.generated.api_pb2_grpc import MlStub
from transport.generated.api_pb2 import PredictRequest, TrainRequest, DataSet, Row, TrainerConfig, ModelConfig, \
    GetModelRequest
import grpc

generate("../impl/test-data/test_1dim.csv", GeneratorConfig(dim=1, rows=200, noize=2, functions=Functions.LINEAR))
s = get_stub()
r = s.train(TrainRequest(
    path="./test-data/test_1dim.csv",
    trainerConfig=TrainerConfig(type="GreedyTrainer", params=[0.01, 100000, 0.001]),
    modelConfig=ModelConfig(type="LinearModel", regularizator="EmptyRegularizator", loss="MSELoss"),
))
print(r)
plot_model_over_dataset("../impl/test-data/test_1dim.csv", r.modelId)
# r = s.getModel(GetModelRequest(id="d3497d95-1040-41a2-a511-e1169e60683a"))
# print(r)
# r = s.predict(PredictRequest(modelId="d1c446b1-b994-4b19-a3fe-751812d73576", data=DataSet(
#     rows=[
#         Row(x=[-0.29451304412658563, -0.49304650406076056]),
#         *[Row(x=[-0.3207816957162245, 0.4676085387753295])] * 100000,
#     ]
# )))
# print(r)
