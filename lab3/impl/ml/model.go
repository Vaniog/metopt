package ml

import "gonum.org/v1/gonum/mat"

type Model interface {
	Predict(mat.Vector) float64
	update(mat.Vector)
	weights() mat.Vector
}
