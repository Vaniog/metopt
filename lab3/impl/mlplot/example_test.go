package mlplot

import (
	"math"
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
			return 2*x*x*x + math.Pow(2*x-0.3, 6) + x
		}, 0), -1, 1, 100), 6)
	// Output:
}
