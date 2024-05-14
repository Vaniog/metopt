package ml

import "gonum.org/v1/gonum/mat"

type Regularizator interface {
	R(weights *mat.VecDense) float64
	Dr(weights *mat.VecDense) *mat.VecDense
}

type EmptyRegularizator struct {
}

func (e EmptyRegularizator) R(_ *mat.VecDense) float64 {
	return 0
}

func (e EmptyRegularizator) Dr(w *mat.VecDense) *mat.VecDense {
	return mat.NewVecDense(w.Len(), nil)
}
