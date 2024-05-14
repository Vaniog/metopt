package ml

import "gonum.org/v1/gonum/mat"

type Row struct {
	X mat.Vector
	Y float64
}

type Dataset interface {
	Row(idx int) Row
	Len() int
}
