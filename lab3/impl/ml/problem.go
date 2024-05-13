package ml

import (
	"gonum.org/v1/gonum/mat"
)

type weights mat.Vector
type ErrorFunc func(Model) float64
type ErrorFuncFabric func(Row) ErrorFunc

//func DefaultErrorFunc(r Row) ErrorFunc {
//	return func(m Model) float64 {
//		return abs(m.Predict(r.X) - r.Y)
//	}
//}

type Problem interface {
	L(idx int) ErrorFunc
	R() ErrorFunc
	Len() int
}

type Model interface {
	update(weights)
	Predict(mat.Vector) float64
}

type Row struct {
	X mat.Vector
	Y float64
}

type Dataset interface {
	Row(idx int) Row
	Len() int
}

type Trainer interface {
	Train(Problem) Model
}

type datasetProblem struct {
	d               Dataset
	errorFuncFabric func(Row) ErrorFunc
	r               ErrorFunc
}

func (d datasetProblem) L(idx int) ErrorFunc {
	//TODO implement me
	panic("implement me")
}

func (d datasetProblem) R() ErrorFunc {
	//TODO implement me
	panic("implement me")
}

func (d datasetProblem) Len() int {
	//TODO implement me
	panic("implement me")
}

func NewProblem(Dataset, func(Row) ErrorFunc, ErrorFunc) Problem {
	return datasetProblem{}
}

func LessSquares(Row) ErrorFunc {
	//TODO implement me
	panic("implement me")
}

var defaultRegularization ErrorFunc

func work() {
	var dataset Dataset
	problem := NewProblem(dataset, LessSquares, defaultRegularization)
	trainer := NewSGDTrainer()
	m := trainer.Train(problem)

	m.Predict(mat.NewVecDense(1, []float64{1}))
}

func NewSGDTrainer() Trainer {
	//TODO implement me
	panic("implement me")
}

// 0.1 0.2 | 3
// Model: (10, 5)
// f: Model -> h_w
// ErrorFunc = f(Model)(x_i) - y_i
