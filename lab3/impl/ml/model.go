package ml

import "gonum.org/v1/gonum/mat"

type Model interface {
	Predict(x mat.Vector) float64
	dw(x mat.Vector) mat.Vector
	weights() *mat.VecDense
}

type LinearModel struct {
	w *mat.VecDense
}

func NewLinearModel(rowLen int) *LinearModel {
	return &LinearModel{
		w: mat.NewVecDense(rowLen, nil),
	}
}

func (lm *LinearModel) dw(x mat.Vector) mat.Vector {
	return x
}

func (lm *LinearModel) Predict(x mat.Vector) float64 {
	return mat.Dot(lm.w, x)
}

func (lm *LinearModel) weights() *mat.VecDense {
	return lm.w
}
