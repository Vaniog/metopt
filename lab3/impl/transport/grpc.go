package transport

import (
	"context"
	"gonum.org/v1/gonum/mat"
	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
	"log"
	"metopt/bencmark"
	"metopt/ml"
	"metopt/ml/training"
	"metopt/transport/generated"
	"net"
)

var ModelsFolder = "./models"

type MlServer struct {
	generated.MlServer
}

func (s *MlServer) Train(ctx context.Context, r *generated.TrainRequest) (*generated.TrainResponse, error) {
	log.Println("start training")
	ds, err := training.NewSliceDatasetFromCSV(r.Path)
	if err != nil {
		return nil, err
	}
	m := ml.NewLinearModel(ml.Config{
		RowLen: ds.Dim(),
		Loss:   ml.MSELoss{},
		Reg:    ml.EmptyRegularizator{},
	})
	trainer := training.NewGreedyTrainer(
		0.001,
		100000,
		0.001,
	)
	_, err, ms := bencmark.Profile(func() (any, error) {
		trainer.Train(m, ds)
		return nil, nil
	})

	if err != nil {
		return nil, err
	}
	serializer := NewModelSerializer(ModelsFolder)
	id, err := serializer.Serialize(m)
	if err != nil {
		return nil, err
	}
	return &generated.TrainResponse{
		ModelId:   id,
		Benchmark: ms,
	}, nil
}

func (s *MlServer) Predict(ctx context.Context, r *generated.PredictRequest) (*generated.PredictResponse, error) {
	serializer := NewModelSerializer(ModelsFolder)
	m, err := serializer.Deserialize(r.ModelId)
	if err != nil {
		return nil, err
	}
	res := make([]float64, len(r.Data.Rows))
	_, err, ms := bencmark.Profile(func() (any, error) {
		for i, row := range r.Data.Rows {
			res[i] = m.Predict(mat.NewVecDense(len(row.X), row.X))
		}
		return nil, nil
	})
	if err != nil {
		return nil, err
	}
	return &generated.PredictResponse{
		Y:         res,
		Benchmark: ms,
	}, nil
}

func StartServer() {
	listener, err := net.Listen("tcp", "localhost:8888")

	if err != nil {
		grpclog.Fatalf("failed to listen: %v", err)
	}

	var opts []grpc.ServerOption
	grpcServer := grpc.NewServer(opts...)

	generated.RegisterMlServer(grpcServer, &MlServer{})
	err = grpcServer.Serve(listener)
	if err != nil {
		grpclog.Fatalf("failed to serve: %v", err)
	}
}