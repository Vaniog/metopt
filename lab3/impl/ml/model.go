package ml

import (
	"gonum.org/v1/gonum/mat"
)

type Model interface {
	Predict(x mat.Vector) float64
	Config() Config
	Weights() *mat.VecDense
	DP(x mat.Vector) mat.Vector
}

type Config struct {
	// RowLen is inputs len
	RowLen int
	Loss   Loss
	Reg    Regularizator
}
