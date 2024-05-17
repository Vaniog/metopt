package mlplot

import (
	"math/rand/v2"
	"metopt/ml/training"
)

func ExamplePredictWithLinearAndPlot() {
	PredictWithLinearAndPlot(training.NewSliceDataSet([][]float64{
		{1, 2},
		{2, 4},
		{3, 6},
		{4, 8},
		{5, 10},
	}))
	// Output:
}

func ExamplePredictWithPolynomialAndPlot() {
	PredictWithPolynomialAndPlot(DatasetFromFunction(
		AppendNoise(func(x float64) float64 {
			return rand.Float64()
		}, 0.2), -1, 1, 4), 5)
	// Output:
}
