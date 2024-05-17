package mlplot

import (
	"math/rand"
	"metopt/ml/training"
)

func DatasetFromFunction(f XYFunc, from, to float64, n int) training.DataSet {
	var data [][]float64
	for range n {
		x := from + (to-from)*rand.Float64()
		data = append(data, []float64{x, f(x)})
	}
	return training.NewSliceDataSet(data)
}
