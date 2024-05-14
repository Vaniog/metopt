package ml

import (
	"gonum.org/v1/gonum/mat"
)

type LinearModel struct {
	w *mat.VecDense
	c Config
}

func (lm *LinearModel) Config() Config {
	return lm.c
}

func NewLinearModel(c Config) *LinearModel {
	return &LinearModel{
		w: mat.NewVecDense(c.RowLen, nil),
		c: c,
	}
}

func (lm *LinearModel) DP(x mat.Vector) mat.Vector {
	return x
}

func (lm *LinearModel) Predict(x mat.Vector) float64 {
	return mat.Dot(lm.w, x)
}

func (lm *LinearModel) Weights() *mat.VecDense {
	return lm.w
}
