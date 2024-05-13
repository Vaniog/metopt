package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func main() {
	v := mat.NewVecDense(2, []float64{2, 1})
	fmt.Println(mat.Formatted(v.T()))
}
