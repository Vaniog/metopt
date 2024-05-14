package ml

import "gonum.org/v1/gonum/mat"

type Loss interface {
	F(predicted, actual float64) float64
	Df(predicted, actual float64) float64
}

type Regularizator interface {
	R(weights mat.Vector) float64
	Dr(weights mat.Vector) mat.Vector
}
