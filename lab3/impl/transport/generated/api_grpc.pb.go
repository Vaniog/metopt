// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.2.0
// - protoc             v4.23.2
// source: api.proto

package generated

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// MlClient is the client API for Ml service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type MlClient interface {
	Train(ctx context.Context, in *TrainRequest, opts ...grpc.CallOption) (*TrainResponse, error)
	Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictResponse, error)
}

type mlClient struct {
	cc grpc.ClientConnInterface
}

func NewMlClient(cc grpc.ClientConnInterface) MlClient {
	return &mlClient{cc}
}

func (c *mlClient) Train(ctx context.Context, in *TrainRequest, opts ...grpc.CallOption) (*TrainResponse, error) {
	out := new(TrainResponse)
	err := c.cc.Invoke(ctx, "/Ml/train", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *mlClient) Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictResponse, error) {
	out := new(PredictResponse)
	err := c.cc.Invoke(ctx, "/Ml/predict", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// MlServer is the server API for Ml service.
// All implementations must embed UnimplementedMlServer
// for forward compatibility
type MlServer interface {
	Train(context.Context, *TrainRequest) (*TrainResponse, error)
	Predict(context.Context, *PredictRequest) (*PredictResponse, error)
	mustEmbedUnimplementedMlServer()
}

// UnimplementedMlServer must be embedded to have forward compatible implementations.
type UnimplementedMlServer struct {
}

func (UnimplementedMlServer) Train(context.Context, *TrainRequest) (*TrainResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Train not implemented")
}
func (UnimplementedMlServer) Predict(context.Context, *PredictRequest) (*PredictResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Predict not implemented")
}
func (UnimplementedMlServer) mustEmbedUnimplementedMlServer() {}

// UnsafeMlServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to MlServer will
// result in compilation errors.
type UnsafeMlServer interface {
	mustEmbedUnimplementedMlServer()
}

func RegisterMlServer(s grpc.ServiceRegistrar, srv MlServer) {
	s.RegisterService(&Ml_ServiceDesc, srv)
}

func _Ml_Train_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(TrainRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MlServer).Train(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/Ml/train",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MlServer).Train(ctx, req.(*TrainRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Ml_Predict_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PredictRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MlServer).Predict(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/Ml/predict",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MlServer).Predict(ctx, req.(*PredictRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Ml_ServiceDesc is the grpc.ServiceDesc for Ml service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Ml_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "Ml",
	HandlerType: (*MlServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "train",
			Handler:    _Ml_Train_Handler,
		},
		{
			MethodName: "predict",
			Handler:    _Ml_Predict_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "api.proto",
}
